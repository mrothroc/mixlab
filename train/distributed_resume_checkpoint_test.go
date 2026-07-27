package train

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

type fakeDistributedCheckpointControl struct {
	rank       int
	world      int
	view       distributed.LocalGroupView
	broadcasts [][]int32
	returns    [][]int32
}

func (f *fakeDistributedCheckpointControl) Rank() int {
	return f.rank
}

func (f *fakeDistributedCheckpointControl) WorldSize() int {
	return f.world
}

func (f *fakeDistributedCheckpointControl) LocalView() distributed.LocalGroupView {
	return f.view
}

func (f *fakeDistributedCheckpointControl) BroadcastControl(
	_ int,
	values []int32,
) ([]int32, error) {
	f.broadcasts = append(f.broadcasts, append([]int32(nil), values...))
	if index := len(f.broadcasts) - 1; index < len(f.returns) {
		return append([]int32(nil), f.returns[index]...), nil
	}
	return append([]int32(nil), values...), nil
}

func TestDistributedResumeManifestRoundTrip(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 10
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	dir := t.TempDir()
	shard := filepath.Join(dir, "train_000.bin")
	if err := os.WriteFile(shard, []byte("distributed dataset"), 0o644); err != nil {
		t.Fatal(err)
	}
	membership, err := distributed.NewDDPGroupMembership(
		"run-1",
		"workers",
		7,
		"ring",
		[]distributed.DDPGroupMember{{MemberID: "rank-zero", Rank: 0}},
	)
	if err != nil {
		t.Fatal(err)
	}
	view, err := distributed.NewLocalGroupView(
		membership,
		"rank-zero",
		0,
		"launch-2",
	)
	if err != nil {
		t.Fatal(err)
	}
	control := &fakeDistributedCheckpointControl{
		rank:  0,
		world: 1,
		view:  view,
	}
	spec := gpu.TrainerOptimizerSpec{
		Groups: []gpu.OptimizerGroup{{
			Kind:    gpu.OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.99,
			Epsilon: 1e-8,
		}},
		Weights:       make([]gpu.WeightOptimizer, len(shapes)),
		DefaultBaseLR: 1e-3,
	}
	state := gpu.TrainerStateSnapshot{
		Optimizer: gpu.TrainerOptimizerStats{
			AttemptedSteps: 4,
			CommittedSteps: 3,
			SkippedSteps:   1,
		},
		Tensors: []gpu.TrainerOptimizerStateTensor{
			{
				Kind:        gpu.OptimizerStateAdamM,
				WeightIndex: 0,
				Shape:       append([]int(nil), shapes[0].Shape...),
				Data:        make([]float32, shapeProduct(shapes[0].Shape)),
			},
			{
				Kind:        gpu.OptimizerStateAdamV,
				WeightIndex: 0,
				Shape:       append([]int(nil), shapes[0].Shape...),
				Data:        make([]float32, shapeProduct(shapes[0].Shape)),
			},
		},
	}
	state.Tensors[0].Data[0] = 0.25
	state.Tensors[1].Data[0] = 0.5
	scheduler, steps := buildTrainingScheduler(cfg.Training)
	schedule, err := resumeScheduleFrom(cfg.Training, scheduler, steps)
	if err != nil {
		t.Fatal(err)
	}
	trainer := fakeResumeTrainer{
		fakeWeightReader: fakeWeightReader{weights: weights},
		state:            state,
		spec:             spec,
	}
	artifacts, manifestPath, err := writeDistributedResumableCheckpoint(
		cfg,
		trainer,
		shapes,
		dir,
		distributedResumableCheckpointContext{
			Control:                 control,
			TrainPattern:            filepath.Join(dir, "train_*.bin"),
			Program:                 program,
			Schedule:                schedule,
			LocalBatchTokens:        16,
			AccumulationSteps:       2,
			Sampler:                 distributedResumeSamplerState{Epoch: 1, LocalBatchCursor: 8, LocalMicrostepsConsumed: 8},
			EffectiveGlobalTokens:   128,
			EffectiveGlobalExamples: 8,
		},
	)
	if err != nil {
		t.Fatalf("writeDistributedResumableCheckpoint: %v", err)
	}
	if got, want := control.broadcasts, [][]int32{{1}, {0, 4, 0}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("broadcasts=%v want=%v", got, want)
	}
	for _, path := range []string{
		artifacts.FinalPath,
		manifestPath,
		filepath.Join(dir, distributedResumeStateFilename(4)),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
	}
	resolved, err := resolveDistributedResumeManifest(dir)
	if err != nil {
		t.Fatalf("resolveDistributedResumeManifest: %v", err)
	}
	loaded, err := loadDistributedResumeState(resolved)
	if err != nil {
		t.Fatalf("loadDistributedResumeState: %v", err)
	}
	if loaded.Manifest.GlobalOptimizerAttempt != 4 ||
		loaded.Manifest.GlobalCommittedStep != 3 ||
		loaded.Manifest.Topology.MembershipGeneration != 7 ||
		loaded.Manifest.Topology.LaunchAttemptID != "launch-2" ||
		loaded.Manifest.LocalBatchTokens != 16 ||
		loaded.Manifest.AccumulationSteps != 2 ||
		loaded.Manifest.Sampler.LocalMicrostepsConsumed != 8 ||
		loaded.ModelPath != artifacts.FinalPath {
		t.Fatalf("unexpected distributed manifest: %+v", loaded.Manifest)
	}
	if loaded.Manifest.ProgramHash == "" ||
		loaded.Manifest.WeightLayoutHash == "" ||
		loaded.Manifest.ConfigHash == "" ||
		loaded.Manifest.OptimizerHash == "" ||
		loaded.Manifest.DatasetHash == "" {
		t.Fatalf("distributed manifest is missing identity hashes: %+v", loaded.Manifest)
	}
	if !reflect.DeepEqual(loaded.Trainer, state) {
		t.Fatalf(
			"distributed trainer state round trip mismatch\n got=%+v\nwant=%+v",
			loaded.Trainer,
			state,
		)
	}
}

func TestDistributedResumeManifestIgnoresUncommittedStagingArtifacts(t *testing.T) {
	dir := t.TempDir()
	stage := filepath.Join(dir, ".mixlab-ddp-checkpoint-partial")
	if err := os.MkdirAll(stage, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(stage, distributedResumeModelFilename(9)),
		[]byte("partial"),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveDistributedResumeManifest(dir); err == nil {
		t.Fatal("uncommitted staging artifacts resolved as a checkpoint")
	}
}

func TestDistributedResumeNonzeroRankPerformsNoFilesystemWrites(t *testing.T) {
	membership, err := distributed.NewDDPGroupMembership(
		"run-1",
		"workers",
		0,
		"ring",
		[]distributed.DDPGroupMember{
			{MemberID: "rank-zero", Rank: 0},
			{MemberID: "rank-one", Rank: 1},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	view, err := distributed.NewLocalGroupView(
		membership,
		"rank-one",
		1,
		"launch-1",
	)
	if err != nil {
		t.Fatal(err)
	}
	control := &fakeDistributedCheckpointControl{
		rank:  1,
		world: 2,
		view:  view,
		returns: [][]int32{
			{1},
			{0, 4, 0},
		},
	}
	dir := filepath.Join(t.TempDir(), "checkpoint")
	artifacts, manifestPath, err := writeDistributedResumableCheckpoint(
		nil,
		nil,
		nil,
		dir,
		distributedResumableCheckpointContext{
			Control:           control,
			AccumulationSteps: 2,
			Sampler: distributedResumeSamplerState{
				LocalMicrostepsConsumed: 8,
			},
		},
	)
	if err != nil {
		t.Fatalf("writeDistributedResumableCheckpoint: %v", err)
	}
	if got, want := control.broadcasts, [][]int32{{1}, {0, 0, 0}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("broadcasts=%v want=%v", got, want)
	}
	if artifacts.FinalPath != filepath.Join(dir, distributedResumeModelFilename(4)) ||
		manifestPath != filepath.Join(dir, distributedResumeManifestFilename(4)) {
		t.Fatalf("unexpected rank-one paths: artifacts=%+v manifest=%s", artifacts, manifestPath)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("nonzero rank created checkpoint directory: %v", err)
	}
}

func TestDistributedResumeRejectsTopologyMismatch(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 10
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	membership, err := distributed.NewDDPGroupMembership(
		"resume-run",
		"workers",
		3,
		"ring",
		[]distributed.DDPGroupMember{{MemberID: "rank-zero", Rank: 0}},
	)
	if err != nil {
		t.Fatal(err)
	}
	view, err := distributed.NewLocalGroupView(
		membership,
		"rank-zero",
		0,
		"first-launch",
	)
	if err != nil {
		t.Fatal(err)
	}
	control := &fakeDistributedCheckpointControl{rank: 0, world: 1, view: view}
	spec := gpu.TrainerOptimizerSpec{
		Groups: []gpu.OptimizerGroup{{
			Kind: gpu.OptimizerAdamW, LR: 1e-3, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8,
		}},
		Weights:       make([]gpu.WeightOptimizer, len(shapes)),
		DefaultBaseLR: 1e-3,
	}
	state := gpu.TrainerStateSnapshot{
		Optimizer: gpu.TrainerOptimizerStats{AttemptedSteps: 2, CommittedSteps: 2},
		Tensors: []gpu.TrainerOptimizerStateTensor{{
			Kind:        gpu.OptimizerStateAdamM,
			WeightIndex: 0,
			Shape:       append([]int(nil), shapes[0].Shape...),
			Data:        make([]float32, shapeProduct(shapes[0].Shape)),
		}},
	}
	trainer := fakeResumeTrainer{
		fakeWeightReader: fakeWeightReader{weights: weights},
		state:            state,
		spec:             spec,
	}
	dir := t.TempDir()
	_, manifestPath, err := writeDistributedResumableCheckpoint(
		cfg,
		trainer,
		shapes,
		dir,
		distributedResumableCheckpointContext{
			Control:           control,
			DatasetHash:       "dataset-fixture",
			Program:           program,
			Schedule:          resumeSchedule{Kind: "cosine", OriginalTotalSteps: 10, ExtensionPolicy: "original_then_floor"},
			LocalBatchTokens:  16,
			AccumulationSteps: 1,
			Sampler: distributedResumeSamplerState{
				LocalBatchCursor:        2,
				LocalMicrostepsConsumed: 2,
			},
		},
	)
	if err != nil {
		t.Fatalf("write distributed checkpoint: %v", err)
	}

	relaunchedView, err := distributed.NewLocalGroupView(
		membership,
		"rank-zero",
		0,
		"second-launch",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareDistributedResume(
		manifestPath,
		cfg,
		program,
		shapes,
		spec,
		&DistributedTrainerContext{
			LocalView:         relaunchedView,
			AccumulationSteps: 1,
			DatasetHash:       "dataset-fixture",
		},
		16,
	); err != nil {
		t.Fatalf("new launch attempt should preserve exact resume: %v", err)
	}

	changedMembership, err := distributed.NewDDPGroupMembership(
		"resume-run",
		"workers",
		4,
		"ring",
		[]distributed.DDPGroupMember{{MemberID: "rank-zero", Rank: 0}},
	)
	if err != nil {
		t.Fatal(err)
	}
	changedView, err := distributed.NewLocalGroupView(
		changedMembership,
		"rank-zero",
		0,
		"third-launch",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareDistributedResume(
		manifestPath,
		cfg,
		program,
		shapes,
		spec,
		&DistributedTrainerContext{
			LocalView:         changedView,
			AccumulationSteps: 1,
			DatasetHash:       "dataset-fixture",
		},
		16,
	); err == nil {
		t.Fatal("changed membership generation was accepted for exact resume")
	}

	warmStart, err := loadDistributedWeightsOnlyWarmStart(manifestPath, shapes)
	if err != nil {
		t.Fatalf("weights-only warm start under changed topology: %v", err)
	}
	if !reflect.DeepEqual(warmStart, weights) {
		t.Fatal("weights-only warm start changed replicated model weights")
	}
}

func TestFirstNonFiniteWeightIndex(t *testing.T) {
	weights := [][]float32{
		{1, 2},
		{3, float32(math.Inf(1))},
		{float32(math.NaN())},
		{4, 5},
	}
	if got := firstNonFiniteWeightIndex(weights); got != 1 {
		t.Fatalf("first non-finite tensor=%d, want 1", got)
	}
	if got := firstNonFiniteWeightIndex([][]float32{{1}, {2}}); got != -1 {
		t.Fatalf("finite weights reported tensor %d", got)
	}

	state := gpu.TrainerStateSnapshot{
		Tensors: []gpu.TrainerOptimizerStateTensor{
			{Data: []float32{1, 2}},
			{Data: []float32{3, float32(math.NaN())}},
			{Data: []float32{float32(math.Inf(-1))}},
		},
	}
	if got := firstNonFiniteTrainerStateIndex(state); got != 1 {
		t.Fatalf("first non-finite optimizer tensor=%d, want 1", got)
	}
}
