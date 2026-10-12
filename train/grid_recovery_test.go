package train

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

func TestGridRecoveryPublicationAndFailure(t *testing.T) {
	cfg := smallSWAArtifactConfig()
	cfg.Training.Steps = 10
	shapes, weights := smallSWAArtifactWeights(t, cfg)
	spec := gpu.TrainerOptimizerSpec{Groups: []gpu.OptimizerGroup{{Kind: gpu.OptimizerAdamW, LR: 1e-3}}, Weights: make([]gpu.WeightOptimizer, len(shapes))}
	sampler, _ := data.NewGridSampler(8, 42)
	_, _, _, _ = sampler.Next(2)
	state := gpu.TrainerStateSnapshot{Optimizer: gpu.TrainerOptimizerStats{AttemptedSteps: 1, CommittedSteps: 1}}
	trainer := fakeResumeTrainer{fakeWeightReader: fakeWeightReader{weights: weights}, state: state, spec: spec}
	sched, steps := buildTrainingScheduler(cfg.Training)
	schedule, _ := resumeScheduleFrom(cfg.Training, sched, steps)
	ctx := resumableCheckpointContext{DatasetHash: "synthetic", Schedule: schedule, Grid: snapshotGridRun(sampler, shapes, .5, TrainResult{HasValLoss: true, LastValLoss: .25}, 2, false)}
	ctx.Grid.ValidationPending = true
	r := newGridRecovery(t.TempDir())
	write := func(dir string) (string, error) {
		_, path, err := writeResumableCheckpoint(cfg, trainer, shapes, dir, 1, ctx)
		return path, err
	}
	if err := r.publish(1, write); err != nil {
		t.Fatal(err)
	}
	first := r.manifest
	m, err := readResumeManifest(first)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := loadResumeState(m)
	if err != nil || loaded.Trainer.Optimizer != state.Optimizer || len(loaded.Trainer.Tensors) != 0 || !reflect.DeepEqual(m.Grid, ctx.Grid) {
		t.Fatal("recovery did not preserve optimizer/sampler/validation", err)
	}
	for _, phase := range []string{"training submit", "training collect", "validation"} {
		cause := errors.New(phase + ": out of memory")
		got := r.failure(cause)
		if !errors.Is(got, cause) || !strings.Contains(got.Error(), first) || !strings.Contains(got.Error(), "completed step 1") {
			t.Fatal(got)
		}
	}
	if _, err := os.Stat(filepath.Join(r.dir, "failure.json")); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"write", "missing-manifest", "wrong-step", "pointer"} {
		t.Run(kind, func(t *testing.T) {
			writer := write
			step := 1
			switch kind {
			case "write":
				writer = func(string) (string, error) { return "", errors.New("disk full") }
			case "missing-manifest":
				writer = func(dir string) (string, error) { return filepath.Join(dir, "absent"), nil }
			case "wrong-step":
				step = 2
			case "pointer":
				pointer := filepath.Join(r.dir, "latest.json")
				if err := os.Remove(pointer); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(pointer, 0o755); err != nil {
					t.Fatal(err)
				}
				defer func() { _ = os.Remove(pointer) }()
			}
			if err := r.publish(step, writer); err == nil {
				t.Fatal("accepted incomplete publication")
			}
			if r.manifest != first {
				t.Fatal("changed known-good pointer")
			}
			if _, err := os.Stat(first); err != nil {
				t.Fatal("deleted known-good bundle", err)
			}
			pending, _ := filepath.Glob(filepath.Join(r.dir, ".pending-*"))
			if len(pending) != 0 {
				t.Fatal("staging leak", pending)
			}
		})
	}
	// Restore the pointer removed by the fault injection; the next successful
	// publication removes the prior bundle only after committing its replacement.
	if err := atomicWriteJSON(filepath.Join(r.dir, "latest.json"), gridRecoveryPointer{Bundle: filepath.Base(filepath.Dir(first)), Step: 1}); err != nil {
		t.Fatal(err)
	}
	if err := r.publish(1, write); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(first); !os.IsNotExist(err) {
		t.Fatal("old bundle retained", err)
	}
	bundles, _ := filepath.Glob(filepath.Join(r.dir, "bundle-*"))
	if len(bundles) != 1 {
		t.Fatal("unbounded recovery retention", bundles)
	}
}

func TestGridRecoveryDisabledAndCadence(t *testing.T) {
	r := newGridRecovery("")
	if err := r.publish(0, func(string) (string, error) { t.Fatal("disabled recovery called writer"); return "", nil }); err != nil {
		t.Fatal(err)
	}
	cause := errors.New("unrelated error")
	if r.failure(cause) != cause || r.failure(nil) != nil {
		t.Fatal("changed non-memory failure")
	}
	if !strings.Contains(r.failure(errors.New("out of memory")).Error(), "no recovery bundle") {
		t.Fatal("missing guidance")
	}
	if gridRecoveryEvery(7, 2) != 7 || gridRecoveryEvery(0, 2) != 2 || gridRecoveryEvery(0, 0) != 1 {
		t.Fatal("wrong recovery cadence")
	}
}
