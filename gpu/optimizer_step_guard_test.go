//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func optimizerGuardBackwardProgram(t *testing.T) *Program {
	t.Helper()
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{1, 1})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Sqrt("w0", "root")
	prog.Mul("root", "x", "scaled")
	prog.MeanAxis("scaled", 0, "mean0")
	prog.MeanAxis("mean0", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	return gpuProg
}

func optimizerGuardLinearProgram(t *testing.T) *Program {
	t.Helper()
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{1, 1})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Mul("w0", "x", "scaled")
	prog.MeanAxis("scaled", 0, "mean0")
	prog.MeanAxis("mean0", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	return gpuProg
}

func optimizerGuardTrainer(t *testing.T, prog *Program, initial float32, group OptimizerGroup, decay bool) TrainerHandle {
	t.Helper()
	weight, err := FromDataShape([]float32{initial}, []int{1, 1})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	t.Cleanup(func() { FreeHandle(weight) })
	trainer, err := CreateTrainer(prog, []int64{weight}, TrainerOptimizerSpec{
		Groups:        []OptimizerGroup{group},
		Weights:       []WeightOptimizer{{GroupIndex: 0, Decay: decay}},
		MaxGradNorm:   1,
		DefaultBaseLR: group.LR,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	t.Cleanup(func() { TrainerDestroy(trainer) })
	return trainer
}

func optimizerGuardInput(value float32) []TensorInput {
	return []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{1, 1}, Data: []float32{value},
	}}
}

func readOptimizerGuardWeight(t *testing.T, trainer TrainerHandle) float32 {
	t.Helper()
	values := make([]float32, 1)
	if err := TrainerReadWeight(trainer, 0, values); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}
	return values[0]
}

func TestOptimizerStepGuardRollsBackNonFiniteGradientAndBiasStep(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := optimizerGuardBackwardProgram(t)
	defer prog.Destroy()
	for _, tc := range []struct {
		name  string
		group OptimizerGroup
	}{
		{name: "adamw", group: OptimizerGroup{Kind: OptimizerAdamW, LR: 0.01, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-6}},
		{name: "lamb", group: OptimizerGroup{Kind: OptimizerLAMB, LR: 0.01, Beta1: 0.9, Beta2: 0.99, Epsilon: 1e-6, LAMBTrustRatioCap: 10}},
		{name: "muon", group: OptimizerGroup{Kind: OptimizerMuon, LR: 0.01, Beta1: 0.9, BackendSteps: 5}},
		{name: "sgd", group: OptimizerGroup{Kind: OptimizerSGD, LR: 0.01, Beta1: 0.9}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			trainer := optimizerGuardTrainer(t, prog, 0, tc.group, false)
			loss, err := TrainerStep(trainer, optimizerGuardInput(0))
			if err != nil {
				t.Fatalf("bad-gradient TrainerStep: %v", err)
			}
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				t.Fatalf("reported skipped-step loss=%g, want finite", loss)
			}
			if got := readOptimizerGuardWeight(t, trainer); got != 0 {
				t.Fatalf("weight after skipped step=%g, want unchanged 0", got)
			}
			stats, err := TrainerOptimizerStatsSnapshot(trainer)
			if err != nil {
				t.Fatalf("TrainerOptimizerStatsSnapshot: %v", err)
			}
			if stats.AttemptedSteps != 1 || stats.CommittedSteps != 0 || stats.SkippedSteps != 1 || !stats.LastStepSkipped {
				t.Fatalf("stats after skipped update=%+v", stats)
			}
			if stats.LastGradientNonfinite != 1 || stats.LastStateNonfinite != 0 {
				t.Fatalf("non-finite gradient was amplified into optimizer state: %+v", stats)
			}
			if stats.ConsecutiveSkipped != 1 {
				t.Fatalf("consecutive skips=%d, want 1", stats.ConsecutiveSkipped)
			}

			if err := TrainerSetWeight(trainer, 0, []float32{1}); err != nil {
				t.Fatalf("TrainerSetWeight: %v", err)
			}
			reference := optimizerGuardTrainer(t, prog, 1, tc.group, false)
			if _, err := TrainerStep(trainer, optimizerGuardInput(1)); err != nil {
				t.Fatalf("valid TrainerStep after skip: %v", err)
			}
			if _, err := TrainerStep(reference, optimizerGuardInput(1)); err != nil {
				t.Fatalf("reference TrainerStep: %v", err)
			}
			got := readOptimizerGuardWeight(t, trainer)
			want := readOptimizerGuardWeight(t, reference)
			if diff := math.Abs(float64(got - want)); diff > 1e-6 {
				t.Fatalf("post-skip first valid update=%g reference=%g diff=%g", got, want, diff)
			}
			stats, err = TrainerOptimizerStatsSnapshot(trainer)
			if err != nil {
				t.Fatalf("TrainerOptimizerStatsSnapshot after valid step: %v", err)
			}
			if stats.AttemptedSteps != 2 || stats.CommittedSteps != 1 || stats.SkippedSteps != 1 || stats.ConsecutiveSkipped != 0 || stats.LastStepSkipped {
				t.Fatalf("stats after recovery=%+v", stats)
			}
		})
	}
}

func TestOptimizerStepGuardFailsAfterThreeConsecutiveBadGradients(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := optimizerGuardBackwardProgram(t)
	defer prog.Destroy()
	trainer := optimizerGuardTrainer(t, prog, 0, OptimizerGroup{
		Kind: OptimizerLAMB, LR: 0.01, Beta1: 0.9, Beta2: 0.99,
		Epsilon: 1e-6, LAMBTrustRatioCap: 10,
	}, false)
	for attempt := 1; attempt <= 2; attempt++ {
		if _, err := TrainerStep(trainer, optimizerGuardInput(0)); err != nil {
			t.Fatalf("attempt %d returned early error: %v", attempt, err)
		}
	}
	if _, err := TrainerStep(trainer, optimizerGuardInput(0)); err == nil {
		t.Fatal("third consecutive invalid update returned success")
	}
	if got := readOptimizerGuardWeight(t, trainer); got != 0 {
		t.Fatalf("weight after circuit breaker=%g, want restored 0", got)
	}
	stats, err := TrainerOptimizerStatsSnapshot(trainer)
	if err != nil {
		t.Fatalf("TrainerOptimizerStatsSnapshot: %v", err)
	}
	if stats.AttemptedSteps != 3 || stats.CommittedSteps != 0 || stats.SkippedSteps != 3 || stats.ConsecutiveSkipped != 3 {
		t.Fatalf("circuit-breaker stats=%+v", stats)
	}
	if stats.LastGradientNonfinite != 1 || stats.LastStateNonfinite != 0 {
		t.Fatalf("circuit-breaker attribution=%+v", stats)
	}
}

func TestOptimizerStepGuardRestoresExistingLAMBMoments(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := optimizerGuardBackwardProgram(t)
	defer prog.Destroy()
	group := OptimizerGroup{
		Kind: OptimizerLAMB, LR: 0.01, Beta1: 0.9, Beta2: 0.99,
		Epsilon: 1e-6, LAMBTrustRatioCap: 10,
	}
	guarded := optimizerGuardTrainer(t, prog, 1, group, false)
	reference := optimizerGuardTrainer(t, prog, 1, group, false)
	for name, trainer := range map[string]TrainerHandle{"guarded": guarded, "reference": reference} {
		if _, err := TrainerStep(trainer, optimizerGuardInput(1)); err != nil {
			t.Fatalf("%s warmup step: %v", name, err)
		}
		if err := TrainerSetWeight(trainer, 0, []float32{0}); err != nil {
			t.Fatalf("%s set bad-point weight: %v", name, err)
		}
	}
	if _, err := TrainerStep(guarded, optimizerGuardInput(0)); err != nil {
		t.Fatalf("guarded bad-gradient step: %v", err)
	}
	for name, trainer := range map[string]TrainerHandle{"guarded": guarded, "reference": reference} {
		if err := TrainerSetWeight(trainer, 0, []float32{1}); err != nil {
			t.Fatalf("%s restore comparison weight: %v", name, err)
		}
		if _, err := TrainerStep(trainer, optimizerGuardInput(1)); err != nil {
			t.Fatalf("%s post-skip valid step: %v", name, err)
		}
	}
	got := readOptimizerGuardWeight(t, guarded)
	want := readOptimizerGuardWeight(t, reference)
	if diff := math.Abs(float64(got - want)); diff > 1e-6 {
		t.Fatalf("post-skip LAMB update=%g reference=%g diff=%g; moments were not restored", got, want, diff)
	}
	stats, err := TrainerOptimizerStatsSnapshot(guarded)
	if err != nil {
		t.Fatalf("TrainerOptimizerStatsSnapshot: %v", err)
	}
	if stats.CommittedSteps != 2 || stats.SkippedSteps != 1 || stats.ConsecutiveSkipped != 0 {
		t.Fatalf("guarded LAMB stats=%+v", stats)
	}
}

func TestOptimizerStepGuardRollsBackNonFiniteCandidateUpdate(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := optimizerGuardLinearProgram(t)
	defer prog.Destroy()
	group := OptimizerGroup{
		Kind: OptimizerAdamW, LR: math.MaxFloat32, Beta1: 0.9, Beta2: 0.99,
		Epsilon: 1e-6, WeightDecay: 2,
	}
	trainer := optimizerGuardTrainer(t, prog, 1, group, true)
	loss, err := TrainerStep(trainer, optimizerGuardInput(1))
	if err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}
	if !isFiniteOptimizerGuardValue(loss) {
		t.Fatalf("loss=%g, want finite forward loss", loss)
	}
	if got := readOptimizerGuardWeight(t, trainer); got != 1 {
		t.Fatalf("weight after overflowing candidate=%g, want rollback to 1", got)
	}
	stats, err := TrainerOptimizerStatsSnapshot(trainer)
	if err != nil {
		t.Fatalf("TrainerOptimizerStatsSnapshot: %v", err)
	}
	if stats.CommittedSteps != 0 || stats.SkippedSteps != 1 || stats.LastStateNonfinite == 0 {
		t.Fatalf("overflow candidate stats=%+v", stats)
	}
}

func isFiniteOptimizerGuardValue(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
