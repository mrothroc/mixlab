package train

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

type distributedCheckpointControl interface {
	Rank() int
	WorldSize() int
	LocalView() distributed.LocalGroupView
	BroadcastControl(rootRank int, localValues []int32) ([]int32, error)
}

type distributedResumableCheckpointContext struct {
	Control                  distributedCheckpointControl
	TrainPattern             string
	DatasetHash              string
	Program                  *arch.Program
	Schedule                 resumeSchedule
	EarlyStop                *earlyStopState
	LocalBatchTokens         int
	AccumulationSteps        int
	Sampler                  distributedResumeSamplerState
	EffectiveGlobalTokens    uint64
	EffectiveGlobalExamples  uint64
	ExactSupportedWorldSizes []int
}

func writeDistributedResumableCheckpoint(
	cfg *ArchConfig,
	trainer any,
	shapes []WeightShape,
	dir string,
	ctx distributedResumableCheckpointContext,
) (safetensorsArtifacts, string, error) {
	if ctx.Control == nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint requires collective control",
		)
	}
	if ctx.Control.Rank() < 0 || ctx.Control.Rank() >= ctx.Control.WorldSize() {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint rank %d outside world size %d",
			ctx.Control.Rank(),
			ctx.Control.WorldSize(),
		)
	}
	if ctx.AccumulationSteps <= 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint accumulation steps must be positive",
		)
	}
	if err := checkpointControlBarrier(ctx.Control, 1); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint pre-publication barrier: %w",
			err,
		)
	}

	var (
		artifacts    safetensorsArtifacts
		manifestPath string
		attempt      uint64
		writeErr     error
	)
	if ctx.Control.Rank() == 0 {
		artifacts, manifestPath, writeErr = writeDistributedCheckpointRankZero(
			cfg,
			trainer,
			shapes,
			dir,
			ctx,
		)
		if writeErr == nil {
			var manifest distributedResumeManifest
			manifest, writeErr = readDistributedResumeManifest(manifestPath)
			attempt = manifest.GlobalOptimizerAttempt
		}
	}
	status := int32(0)
	if writeErr != nil {
		status = 1
	}
	observed, syncErr := ctx.Control.BroadcastControl(0, []int32{
		status,
		int32(uint32(attempt)),
		int32(uint32(attempt >> 32)),
	})
	if syncErr != nil {
		return safetensorsArtifacts{}, "", errors.Join(
			writeErr,
			fmt.Errorf("distributed checkpoint completion broadcast: %w", syncErr),
		)
	}
	if len(observed) != 3 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint completion broadcast returned %d values",
			len(observed),
		)
	}
	if observed[0] != 0 {
		if writeErr != nil {
			return safetensorsArtifacts{}, "", writeErr
		}
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"rank zero failed to publish distributed checkpoint",
		)
	}
	attempt = uint64(uint32(observed[1])) |
		uint64(uint32(observed[2]))<<32
	if attempt == 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"rank zero published an invalid distributed checkpoint attempt",
		)
	}
	wantMicrosteps := attempt * uint64(ctx.AccumulationSteps)
	if ctx.Sampler.LocalMicrostepsConsumed != wantMicrosteps {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"rank %d sampler microsteps=%d do not match rank-zero checkpoint attempt %d * accumulation %d",
			ctx.Control.Rank(),
			ctx.Sampler.LocalMicrostepsConsumed,
			attempt,
			ctx.AccumulationSteps,
		)
	}
	if ctx.Control.Rank() != 0 {
		manifestPath = filepath.Join(
			dir,
			distributedResumeManifestFilename(attempt),
		)
		artifacts.FinalPath = filepath.Join(
			dir,
			distributedResumeModelFilename(attempt),
		)
	}
	return artifacts, manifestPath, nil
}

func checkpointControlBarrier(control distributedCheckpointControl, marker int32) error {
	observed, err := control.BroadcastControl(0, []int32{marker})
	if err != nil {
		return err
	}
	if len(observed) != 1 || observed[0] != marker {
		return fmt.Errorf("expected marker %d, got %v", marker, observed)
	}
	return nil
}

func writeDistributedCheckpointRankZero(
	cfg *ArchConfig,
	trainer any,
	shapes []WeightShape,
	dir string,
	ctx distributedResumableCheckpointContext,
) (safetensorsArtifacts, string, error) {
	stateReader, ok := trainer.(gpuTrainerStateReader)
	if !ok {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"trainer does not support resumable optimizer-state checkpoints",
		)
	}
	specReader, ok := trainer.(gpuOptimizerSpecReader)
	if !ok {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"trainer does not expose its optimizer specification",
		)
	}
	if ctx.Program == nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint requires the training program",
		)
	}
	if ctx.LocalBatchTokens <= 0 || ctx.AccumulationSteps <= 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint local batch tokens and accumulation steps must be positive",
		)
	}
	view := ctx.Control.LocalView()
	if ctx.Control.WorldSize() != view.Membership.WorldSize() {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"distributed checkpoint runtime world size=%d membership world size=%d",
			ctx.Control.WorldSize(),
			view.Membership.WorldSize(),
		)
	}

	snapshot, err := stateReader.ReadTrainerState()
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("read optimizer state: %w", err)
	}
	if snapshot.Optimizer.AttemptedSteps == 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"cannot checkpoint before the first optimizer attempt",
		)
	}
	if index := firstNonFiniteTrainerStateIndex(snapshot); index >= 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"optimizer state tensor %d contains non-finite values",
			index,
		)
	}
	wantMicrosteps := snapshot.Optimizer.AttemptedSteps *
		uint64(ctx.AccumulationSteps)
	if ctx.Sampler.LocalMicrostepsConsumed != wantMicrosteps {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"sampler microsteps=%d, want optimizer attempts %d * accumulation %d",
			ctx.Sampler.LocalMicrostepsConsumed,
			snapshot.Optimizer.AttemptedSteps,
			ctx.AccumulationSteps,
		)
	}

	manifest, tensors, err := buildDistributedResumeManifest(
		cfg,
		shapes,
		snapshot,
		specReader.OptimizerSpec(),
		ctx,
		view,
	)
	if err != nil {
		return safetensorsArtifacts{}, "", err
	}
	attempt := snapshot.Optimizer.AttemptedSteps
	manifest.ModelFile = distributedResumeModelFilename(attempt)
	manifest.StateFile = distributedResumeStateFilename(attempt)
	manifestName := distributedResumeManifestFilename(attempt)
	manifestPath := filepath.Join(dir, manifestName)
	finalArtifacts := safetensorsArtifacts{
		FinalPath: filepath.Join(dir, manifest.ModelFile),
	}
	finalStatePath := filepath.Join(dir, manifest.StateFile)
	for _, path := range []string{
		manifestPath,
		finalArtifacts.FinalPath,
		finalStatePath,
	} {
		if _, statErr := os.Stat(path); statErr == nil {
			return safetensorsArtifacts{}, "", fmt.Errorf(
				"refusing to overwrite distributed checkpoint artifact %q",
				path,
			)
		} else if !os.IsNotExist(statErr) {
			return safetensorsArtifacts{}, "", statErr
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return safetensorsArtifacts{}, "", err
	}
	stageDir, err := os.MkdirTemp(dir, ".mixlab-ddp-checkpoint-*")
	if err != nil {
		return safetensorsArtifacts{}, "", err
	}
	defer func() {
		_ = os.RemoveAll(stageDir)
	}()

	stageModelPath := filepath.Join(stageDir, manifest.ModelFile)
	stageStatePath := filepath.Join(stageDir, manifest.StateFile)
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"read trainer weights: %w",
			err,
		)
	}
	if index := firstNonFiniteWeightIndex(weights); index >= 0 {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"model weight tensor %d contains non-finite values",
			index,
		)
	}
	if err := exportSafetensors(stageModelPath, cfg, shapes, weights); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"stage distributed model: %w",
			err,
		)
	}
	if err := writeNamedFloatSafetensorsAtomic(
		stageStatePath,
		tensors,
		map[string]string{
			"format": distributedResumeCheckpointFormat,
			"step":   fmt.Sprintf("%d", attempt),
		},
	); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"stage distributed optimizer state: %w",
			err,
		)
	}

	moved := make([]string, 0, 2)
	committed := false
	defer func() {
		if !committed {
			for _, path := range moved {
				_ = os.Remove(path)
			}
		}
	}()
	for _, move := range []struct {
		from string
		to   string
	}{
		{from: stageModelPath, to: finalArtifacts.FinalPath},
		{from: stageStatePath, to: finalStatePath},
	} {
		if err := os.Rename(move.from, move.to); err != nil {
			return safetensorsArtifacts{}, "", fmt.Errorf(
				"commit distributed checkpoint artifact %q: %w",
				move.to,
				err,
			)
		}
		moved = append(moved, move.to)
	}
	manifest.CheckpointSizeBytes =
		fileSizeOrZero(finalArtifacts.FinalPath) + fileSizeOrZero(finalStatePath)
	if err := atomicWriteJSON(manifestPath, manifest); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf(
			"publish distributed resume manifest: %w",
			err,
		)
	}
	committed = true
	return finalArtifacts, manifestPath, nil
}

func buildDistributedResumeManifest(
	cfg *ArchConfig,
	shapes []WeightShape,
	snapshot gpu.TrainerStateSnapshot,
	optimizer gpu.TrainerOptimizerSpec,
	ctx distributedResumableCheckpointContext,
	view distributed.LocalGroupView,
) (distributedResumeManifest, []namedFloatTensor, error) {
	configHash, err := resumeConfigHash(cfg)
	if err != nil {
		return distributedResumeManifest{}, nil, fmt.Errorf(
			"hash config: %w",
			err,
		)
	}
	programHash, err := hashJSONHex(ctx.Program)
	if err != nil {
		return distributedResumeManifest{}, nil, fmt.Errorf(
			"hash training program: %w",
			err,
		)
	}
	weightLayoutHash, err := hashJSONHex(shapes)
	if err != nil {
		return distributedResumeManifest{}, nil, fmt.Errorf(
			"hash weight layout: %w",
			err,
		)
	}
	optimizerHash, err := optimizerSpecHash(optimizer)
	if err != nil {
		return distributedResumeManifest{}, nil, fmt.Errorf(
			"hash optimizer plan: %w",
			err,
		)
	}
	datasetHash := ctx.DatasetHash
	if ctx.TrainPattern != "" {
		calculatedHash, hashErr := trainingDatasetHash(ctx.TrainPattern)
		if hashErr != nil {
			return distributedResumeManifest{}, nil, fmt.Errorf(
				"hash training dataset: %w",
				hashErr,
			)
		}
		if datasetHash != "" && datasetHash != calculatedHash {
			return distributedResumeManifest{}, nil, fmt.Errorf(
				"distributed dataset hash=%s does not match training shards=%s",
				datasetHash,
				calculatedHash,
			)
		}
		datasetHash = calculatedHash
	}
	if datasetHash == "" {
		return distributedResumeManifest{}, nil, fmt.Errorf(
			"distributed checkpoint requires a dataset hash or training shard pattern",
		)
	}
	exactWorldSizes := append([]int(nil), ctx.ExactSupportedWorldSizes...)
	if len(exactWorldSizes) == 0 {
		exactWorldSizes = []int{view.Membership.WorldSize()}
	}
	sort.Ints(exactWorldSizes)
	exactWorldSizes = slices.Compact(exactWorldSizes)
	manifest := distributedResumeManifest{
		Format:                 distributedResumeCheckpointFormat,
		GlobalOptimizerAttempt: snapshot.Optimizer.AttemptedSteps,
		GlobalCommittedStep:    snapshot.Optimizer.CommittedSteps,
		Topology: distributedResumeTopology{
			RunID:                view.Membership.RunID,
			GroupID:              view.Membership.GroupID,
			MembershipGeneration: view.Membership.Generation,
			Backend:              view.Membership.Backend,
			OrderedMembers: append(
				[]distributed.DDPGroupMember(nil),
				view.Membership.OrderedMembers...,
			),
			MembersHash:     view.Membership.MembersHash,
			LaunchAttemptID: view.LaunchAttemptID,
		},
		LocalBatchTokens:         ctx.LocalBatchTokens,
		AccumulationSteps:        ctx.AccumulationSteps,
		EffectiveGlobalTokens:    ctx.EffectiveGlobalTokens,
		EffectiveGlobalExamples:  ctx.EffectiveGlobalExamples,
		ConfigHash:               configHash,
		ProgramHash:              programHash,
		WeightLayoutHash:         weightLayoutHash,
		OptimizerHash:            optimizerHash,
		DatasetHash:              datasetHash,
		TrainPattern:             ctx.TrainPattern,
		Sampler:                  ctx.Sampler,
		Schedule:                 ctx.Schedule,
		Optimizer:                snapshot.Optimizer,
		EarlyStop:                ctx.EarlyStop.resumeSnapshot(),
		ExactSupportedWorldSizes: exactWorldSizes,
	}
	tensors := make(
		[]namedFloatTensor,
		0,
		max(1, len(snapshot.Tensors)),
	)
	for _, state := range snapshot.Tensors {
		name := optimizerStateTensorName(state.Kind, state.WeightIndex)
		shape := append([]int(nil), state.Shape...)
		tensors = append(tensors, namedFloatTensor{
			Name:  name,
			Shape: shape,
			Data:  state.Data,
		})
		manifest.OptimizerTensors = append(
			manifest.OptimizerTensors,
			resumeTensorRef{
				Name:        name,
				Kind:        state.Kind,
				WeightIndex: state.WeightIndex,
				Shape:       shape,
			},
		)
	}
	if len(tensors) == 0 {
		tensors = append(tensors, namedFloatTensor{
			Name:  "resume_state",
			Shape: []int{1},
			Data:  []float32{float32(snapshot.Optimizer.AttemptedSteps)},
		})
	}
	return manifest, tensors, nil
}

func loadDistributedResumeState(
	manifest distributedResumeManifest,
) (distributedResumeLoadedState, error) {
	dir := filepath.Dir(manifest.ManifestPath)
	blobs, err := loadSafetensors(filepath.Join(dir, manifest.StateFile))
	if err != nil {
		return distributedResumeLoadedState{}, err
	}
	trainer := gpu.TrainerStateSnapshot{
		Optimizer: manifest.Optimizer,
		Tensors: make(
			[]gpu.TrainerOptimizerStateTensor,
			0,
			len(manifest.OptimizerTensors),
		),
	}
	for _, ref := range manifest.OptimizerTensors {
		data, err := decodeSafetensorFloat32(ref.Name, ref.Shape, blobs)
		if err != nil {
			return distributedResumeLoadedState{}, err
		}
		trainer.Tensors = append(
			trainer.Tensors,
			gpu.TrainerOptimizerStateTensor{
				Kind:        ref.Kind,
				WeightIndex: ref.WeightIndex,
				Shape:       append([]int(nil), ref.Shape...),
				Data:        data,
			},
		)
	}
	return distributedResumeLoadedState{
		Manifest:  manifest,
		ModelPath: filepath.Join(dir, manifest.ModelFile),
		Trainer:   trainer,
	}, nil
}

func fileSizeOrZero(path string) int64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
