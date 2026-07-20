package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

type fakeResumeTrainer struct {
	fakeWeightReader
	state gpu.TrainerStateSnapshot
	spec  gpu.TrainerOptimizerSpec
}

func (f fakeResumeTrainer) ReadTrainerState() (gpu.TrainerStateSnapshot, error) {
	return f.state, nil
}

func (f fakeResumeTrainer) OptimizerSpec() gpu.TrainerOptimizerSpec {
	return f.spec
}

func TestResumableCheckpointBundleRoundTripAndNewestComplete(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 10
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	dir := t.TempDir()
	shard := filepath.Join(dir, "train_000.bin")
	if err := os.WriteFile(shard, []byte("dataset identity"), 0o644); err != nil {
		t.Fatal(err)
	}
	pattern := filepath.Join(dir, "train_*.bin")
	spec := gpu.TrainerOptimizerSpec{
		Groups:        []gpu.OptimizerGroup{{Kind: gpu.OptimizerAdamW, LR: 1e-3, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-8}},
		Weights:       make([]gpu.WeightOptimizer, len(shapes)),
		DefaultBaseLR: 1e-3,
	}
	state := gpu.TrainerStateSnapshot{
		Optimizer: gpu.TrainerOptimizerStats{AttemptedSteps: 2, CommittedSteps: 2},
		Tensors: []gpu.TrainerOptimizerStateTensor{
			{Kind: gpu.OptimizerStateAdamM, WeightIndex: 0, Shape: append([]int(nil), shapes[0].Shape...), Data: make([]float32, shapeProduct(shapes[0].Shape))},
			{Kind: gpu.OptimizerStateAdamV, WeightIndex: 0, Shape: append([]int(nil), shapes[0].Shape...), Data: make([]float32, shapeProduct(shapes[0].Shape))},
		},
	}
	state.Tensors[0].Data[0] = 0.25
	state.Tensors[1].Data[0] = 0.5
	scheduler, steps := buildTrainingScheduler(cfg.Training)
	schedule, err := resumeScheduleFrom(cfg.Training, scheduler, steps)
	if err != nil {
		t.Fatal(err)
	}
	trainer := fakeResumeTrainer{fakeWeightReader: fakeWeightReader{weights: weights}, state: state, spec: spec}
	artifacts, manifestPath, err := writeResumableCheckpoint(cfg, trainer, shapes, dir, 2, resumableCheckpointContext{
		TrainPattern: pattern,
		Schedule:     schedule,
	})
	if err != nil {
		t.Fatalf("writeResumableCheckpoint: %v", err)
	}
	for _, path := range []string{artifacts.FinalPath, manifestPath, filepath.Join(dir, resumeStateFilename(2))} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("stat %s: %v", path, err)
		}
	}

	resolved, err := resolveResumeManifest(dir)
	if err != nil {
		t.Fatalf("resolveResumeManifest: %v", err)
	}
	loaded, err := loadResumeState(resolved)
	if err != nil {
		t.Fatalf("loadResumeState: %v", err)
	}
	if loaded.Manifest.GlobalStep != 2 || loaded.ModelPath != artifacts.FinalPath {
		t.Fatalf("loaded manifest=%+v model=%s", loaded.Manifest, loaded.ModelPath)
	}
	if !reflect.DeepEqual(loaded.Trainer, state) {
		t.Fatalf("trainer state round trip mismatch\n got=%+v\nwant=%+v", loaded.Trainer, state)
	}

	// A higher partial checkpoint is ignored because no completion manifest can
	// safely reference all of its companion artifacts.
	if err := os.WriteFile(filepath.Join(dir, "step_000003.state.safetensors"), []byte("partial"), 0o644); err != nil {
		t.Fatal(err)
	}
	resolved, err = resolveResumeManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.GlobalStep != 2 {
		t.Fatalf("resolved step=%d want=2", resolved.GlobalStep)
	}
}

func TestResumeConfigHashAllowsOnlyStandardStepExtension(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 100
	base, err := resumeConfigHash(cfg)
	if err != nil {
		t.Fatal(err)
	}
	extended := *cfg
	extended.Training = cfg.Training
	extended.Training.Steps = 200
	got, err := resumeConfigHash(&extended)
	if err != nil {
		t.Fatal(err)
	}
	if got != base {
		t.Fatalf("steps-only extension changed config hash: %s != %s", got, base)
	}
	extended.Training.LR *= 2
	got, err = resumeConfigHash(&extended)
	if err != nil {
		t.Fatal(err)
	}
	if got == base {
		t.Fatal("learning-rate change did not change resume config hash")
	}
}

func TestResumeScheduleContinuesOriginalHorizonAtFloor(t *testing.T) {
	original := trainingScheduleWithOptions(1e-3, 100, 0, 0.1, trainingScheduleOptions{WarmupSteps: 10, WarmupStepsSet: true, HoldStepsSet: true})
	saved := resumeSchedule{Kind: "cosine", OriginalTotalSteps: 100, Standard: &original, ExtensionPolicy: "original_then_floor"}
	scheduler, total, err := schedulerForResume(saved, 150)
	if err != nil {
		t.Fatal(err)
	}
	if total != 150 {
		t.Fatalf("total=%d want=150", total)
	}
	floor := original.At(100)
	for step := 100; step < 150; step++ {
		if got := scheduler.At(step); got != floor {
			t.Fatalf("At(%d)=%g want floor %g", step, got, floor)
		}
	}
	if _, _, err := schedulerForResume(saved, 99); err == nil {
		t.Fatal("shrinking below original schedule horizon should fail")
	}
	phaseSaved := resumeSchedule{
		Kind:               "phases",
		OriginalTotalSteps: 100,
		Phases:             []TrainingPhase{{Steps: 100, LR: 1e-3}},
		ExtensionPolicy:    "original_then_floor",
	}
	if _, _, err := schedulerForResume(phaseSaved, 101); err == nil {
		t.Fatal("phase-schedule extension should fail")
	}
}

func TestResumeEarlyStopSnapshotBeforeFirstValidationIsJSONSafe(t *testing.T) {
	state := newEarlyStopState(&EarlyStopSpec{Patience: 3})
	snapshot := state.resumeSnapshot()
	if snapshot.HaveBest || snapshot.Best != 0 {
		t.Fatalf("unexpected initial early-stop snapshot: %+v", snapshot)
	}
	if _, err := json.Marshal(snapshot); err != nil {
		t.Fatalf("marshal initial early-stop snapshot: %v", err)
	}
	restored := newEarlyStopState(&EarlyStopSpec{Patience: 3})
	if err := restored.restoreResumeSnapshot(snapshot); err != nil {
		t.Fatalf("restore initial early-stop snapshot: %v", err)
	}
	if restored.haveBest || restored.stale != 0 {
		t.Fatalf("unexpected restored early-stop state: %+v", restored)
	}
}

func TestDropoutKeysAreStepAndOpDeterministic(t *testing.T) {
	a := make([]int32, 8)
	b := make([]int32, 8)
	fillDropoutKeys(a, 17, 23)
	fillDropoutKeys(b, 17, 23)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("same seed/step keys differ: %v vs %v", a, b)
	}
	fillDropoutKeys(b, 17, 24)
	if reflect.DeepEqual(a, b) {
		t.Fatal("different steps produced identical dropout keys")
	}
}
