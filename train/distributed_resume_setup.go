package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

type distributedResumeExpectations struct {
	LocalView         distributed.LocalGroupView
	LocalBatchTokens  int
	AccumulationSteps int
	ConfigHash        string
	ProgramHash       string
	WeightLayoutHash  string
	OptimizerHash     string
	DatasetHash       string
}

// DistributedResumePlan is a validated exact-resume plan for one fixed DDP
// topology. The loaded checkpoint state remains private so callers must use
// the restore/load helpers below rather than bypassing compatibility checks.
type DistributedResumePlan struct {
	loaded                  distributedResumeLoadedState
	StartOptimizerAttempt   uint64
	StartCommittedStep      uint64
	LocalMicrostepsConsumed uint64
}

func newDistributedResumeExpectations(
	cfg *ArchConfig,
	program *arch.Program,
	shapes []WeightShape,
	optimizer gpu.TrainerOptimizerSpec,
	context *DistributedTrainerContext,
	localBatchTokens int,
) (distributedResumeExpectations, error) {
	if cfg == nil || program == nil {
		return distributedResumeExpectations{}, fmt.Errorf(
			"distributed resume requires config and training program",
		)
	}
	if context == nil {
		return distributedResumeExpectations{}, fmt.Errorf(
			"distributed resume requires trainer context",
		)
	}
	view, err := distributed.NewLocalGroupView(
		context.LocalView.Membership,
		context.LocalView.LocalMemberID,
		context.LocalView.LocalRank,
		context.LocalView.LaunchAttemptID,
	)
	if err != nil {
		return distributedResumeExpectations{}, fmt.Errorf(
			"validate distributed resume identity: %w",
			err,
		)
	}
	accumulationSteps := context.AccumulationSteps
	if accumulationSteps == 0 {
		accumulationSteps = 1
	}
	if localBatchTokens <= 0 || accumulationSteps <= 0 {
		return distributedResumeExpectations{}, fmt.Errorf(
			"distributed resume local batch tokens and accumulation must be positive",
		)
	}
	if context.DatasetHash == "" {
		return distributedResumeExpectations{}, fmt.Errorf(
			"distributed resume requires a dataset hash",
		)
	}
	configHash, err := resumeConfigHash(cfg)
	if err != nil {
		return distributedResumeExpectations{}, err
	}
	programHash, err := hashJSONHex(program)
	if err != nil {
		return distributedResumeExpectations{}, err
	}
	weightLayoutHash, err := hashJSONHex(shapes)
	if err != nil {
		return distributedResumeExpectations{}, err
	}
	optimizerHash, err := optimizerSpecHash(optimizer)
	if err != nil {
		return distributedResumeExpectations{}, err
	}
	return distributedResumeExpectations{
		LocalView:         view,
		LocalBatchTokens:  localBatchTokens,
		AccumulationSteps: accumulationSteps,
		ConfigHash:        configHash,
		ProgramHash:       programHash,
		WeightLayoutHash:  weightLayoutHash,
		OptimizerHash:     optimizerHash,
		DatasetHash:       context.DatasetHash,
	}, nil
}

// PrepareDistributedResume validates and loads a distributed checkpoint for
// the supplied immutable topology and trainer layout.
func PrepareDistributedResume(
	path string,
	cfg *ArchConfig,
	program *arch.Program,
	shapes []WeightShape,
	optimizer gpu.TrainerOptimizerSpec,
	context *DistributedTrainerContext,
	localBatchTokens int,
) (DistributedResumePlan, error) {
	expected, err := newDistributedResumeExpectations(
		cfg,
		program,
		shapes,
		optimizer,
		context,
		localBatchTokens,
	)
	if err != nil {
		return DistributedResumePlan{}, err
	}
	manifest, err := resolveDistributedResumeManifest(path)
	if err != nil {
		return DistributedResumePlan{}, err
	}
	if err := validateDistributedResumeCompatibility(manifest, expected); err != nil {
		return DistributedResumePlan{}, err
	}
	loaded, err := loadDistributedResumeState(manifest)
	if err != nil {
		return DistributedResumePlan{}, fmt.Errorf(
			"load distributed resumable checkpoint: %w",
			err,
		)
	}
	if index := firstNonFiniteTrainerStateIndex(loaded.Trainer); index >= 0 {
		return DistributedResumePlan{}, fmt.Errorf(
			"distributed resumable checkpoint optimizer tensor %d contains non-finite values",
			index,
		)
	}
	return DistributedResumePlan{
		loaded:                  loaded,
		StartOptimizerAttempt:   manifest.GlobalOptimizerAttempt,
		StartCommittedStep:      manifest.GlobalCommittedStep,
		LocalMicrostepsConsumed: manifest.Sampler.LocalMicrostepsConsumed,
	}, nil
}

func validateDistributedResumeCompatibility(
	manifest distributedResumeManifest,
	expected distributedResumeExpectations,
) error {
	current := expected.LocalView
	topology := manifest.Topology
	if topology.RunID != current.Membership.RunID ||
		topology.GroupID != current.Membership.GroupID ||
		topology.MembershipGeneration != current.Membership.Generation ||
		topology.Backend != current.Membership.Backend ||
		topology.MembersHash != current.Membership.MembersHash ||
		!reflect.DeepEqual(
			topology.OrderedMembers,
			current.Membership.OrderedMembers,
		) {
		return fmt.Errorf(
			"distributed resume topology mismatch: checkpoint run=%q group=%q generation=%d backend=%q world=%d members=%s; current run=%q group=%q generation=%d backend=%q world=%d members=%s",
			topology.RunID,
			topology.GroupID,
			topology.MembershipGeneration,
			topology.Backend,
			len(topology.OrderedMembers),
			topology.MembersHash,
			current.Membership.RunID,
			current.Membership.GroupID,
			current.Membership.Generation,
			current.Membership.Backend,
			current.Membership.WorldSize(),
			current.Membership.MembersHash,
		)
	}
	worldSupported := false
	for _, worldSize := range manifest.ExactSupportedWorldSizes {
		if worldSize == current.Membership.WorldSize() {
			worldSupported = true
			break
		}
	}
	if !worldSupported {
		return fmt.Errorf(
			"distributed resume world size %d is not supported exactly by checkpoint %v",
			current.Membership.WorldSize(),
			manifest.ExactSupportedWorldSizes,
		)
	}
	if manifest.LocalBatchTokens != expected.LocalBatchTokens ||
		manifest.AccumulationSteps != expected.AccumulationSteps {
		return fmt.Errorf(
			"distributed resume batch topology mismatch: checkpoint local_batch_tokens=%d accumulation=%d; current local_batch_tokens=%d accumulation=%d",
			manifest.LocalBatchTokens,
			manifest.AccumulationSteps,
			expected.LocalBatchTokens,
			expected.AccumulationSteps,
		)
	}
	for _, hash := range []struct {
		name       string
		checkpoint string
		current    string
	}{
		{name: "config", checkpoint: manifest.ConfigHash, current: expected.ConfigHash},
		{name: "program", checkpoint: manifest.ProgramHash, current: expected.ProgramHash},
		{name: "weight layout", checkpoint: manifest.WeightLayoutHash, current: expected.WeightLayoutHash},
		{name: "optimizer", checkpoint: manifest.OptimizerHash, current: expected.OptimizerHash},
		{name: "dataset", checkpoint: manifest.DatasetHash, current: expected.DatasetHash},
	} {
		if hash.checkpoint != hash.current {
			return fmt.Errorf(
				"distributed resume %s hash mismatch: checkpoint=%s current=%s",
				hash.name,
				hash.checkpoint,
				hash.current,
			)
		}
	}
	wantMicrosteps := manifest.GlobalOptimizerAttempt *
		uint64(expected.AccumulationSteps)
	if manifest.Sampler.LocalMicrostepsConsumed != wantMicrosteps {
		return fmt.Errorf(
			"distributed resume is not at an optimizer boundary: local microsteps=%d want=%d",
			manifest.Sampler.LocalMicrostepsConsumed,
			wantMicrosteps,
		)
	}
	return nil
}

// RestoreDistributedResumableTrainerState restores optimizer state after the
// caller has initialized a trainer with LoadDistributedResumeModelWeights.
func RestoreDistributedResumableTrainerState(
	trainer any,
	setup DistributedResumePlan,
) error {
	restorer, ok := trainer.(gpuTrainerStateRestorer)
	if !ok {
		return fmt.Errorf(
			"trainer does not support distributed optimizer-state restore",
		)
	}
	specReader, ok := trainer.(gpuOptimizerSpecReader)
	if !ok {
		return fmt.Errorf("trainer does not expose its optimizer specification")
	}
	currentHash, err := optimizerSpecHash(specReader.OptimizerSpec())
	if err != nil {
		return err
	}
	if currentHash != setup.loaded.Manifest.OptimizerHash {
		return fmt.Errorf(
			"optimizer configuration does not match distributed checkpoint",
		)
	}
	if err := restorer.RestoreTrainerState(setup.loaded.Trainer); err != nil {
		return fmt.Errorf("restore distributed optimizer state: %w", err)
	}
	return nil
}

func loadDistributedWeightsOnlyWarmStart(
	path string,
	shapes []WeightShape,
) ([][]float32, error) {
	if path == "" {
		return nil, fmt.Errorf("weights-only warm start path is required")
	}
	modelPath := path
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() || strings.HasSuffix(path, ".distributed.resume.json") {
		manifest, err := resolveDistributedResumeManifest(path)
		if err != nil {
			return nil, err
		}
		modelPath = filepath.Join(filepath.Dir(manifest.ManifestPath), manifest.ModelFile)
	}
	weights, err := loadSafetensorsWeights(modelPath, shapes)
	if err != nil {
		return nil, fmt.Errorf(
			"load distributed weights-only warm start %q: %w",
			modelPath,
			err,
		)
	}
	if index := firstNonFiniteWeightIndex(weights); index >= 0 {
		return nil, fmt.Errorf(
			"distributed weights-only warm start tensor %d contains non-finite values",
			index,
		)
	}
	return weights, nil
}

func firstNonFiniteWeightIndex(weights [][]float32) int {
	return firstNonFiniteTensorIndex(
		len(weights),
		func(index int) []float32 { return weights[index] },
	)
}

func firstNonFiniteTrainerStateIndex(snapshot gpu.TrainerStateSnapshot) int {
	return firstNonFiniteTensorIndex(
		len(snapshot.Tensors),
		func(index int) []float32 { return snapshot.Tensors[index].Data },
	)
}

func firstNonFiniteTensorIndex(
	count int,
	values func(index int) []float32,
) int {
	if count == 0 {
		return -1
	}
	workers := min(runtime.GOMAXPROCS(0), count)
	var next atomic.Int64
	var first atomic.Int64
	first.Store(int64(count))
	var wg sync.WaitGroup
	wg.Add(workers)
	for range workers {
		go func() {
			defer wg.Done()
			for {
				index := int(next.Add(1) - 1)
				if index >= count || int64(index) >= first.Load() {
					return
				}
				for _, value := range values(index) {
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
						for {
							current := first.Load()
							if int64(index) >= current ||
								first.CompareAndSwap(current, int64(index)) {
								break
							}
						}
						break
					}
				}
			}
		}()
	}
	wg.Wait()
	index := int(first.Load())
	if index == count {
		return -1
	}
	return index
}

// LoadDistributedResumeModelWeights loads the replicated model tensors from a
// validated distributed resume plan.
func LoadDistributedResumeModelWeights(
	setup DistributedResumePlan,
	shapes []WeightShape,
) ([][]float32, error) {
	return loadDistributedWeightsOnlyWarmStart(
		setup.loaded.ModelPath,
		shapes,
	)
}

// ReplayDistributedTrainingLoader restores the deterministic local loader
// position recorded by a validated distributed resume plan.
func ReplayDistributedTrainingLoader(
	loader *data.Loader,
	setup DistributedResumePlan,
	batchTokens, seqLen int,
) error {
	sampler := setup.loaded.Manifest.Sampler
	if sampler.LocalMicrostepsConsumed > uint64(math.MaxInt) {
		return fmt.Errorf(
			"distributed resume microstep count %d exceeds host integer range",
			sampler.LocalMicrostepsConsumed,
		)
	}
	return replayTrainingLoader(
		loader,
		int(sampler.LocalMicrostepsConsumed),
		batchTokens,
		seqLen,
	)
}
