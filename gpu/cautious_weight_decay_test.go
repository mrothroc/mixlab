//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func cautiousWeightDecayProgram(t *testing.T) *Program {
	t.Helper()
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2, 2})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Mul("w0", "x", "prod")
	prog.MeanAxis("prod", 0, "mean0")
	prog.MeanAxis("mean0", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	return gpuProg
}

func cautiousWeightDecayTrainer(t *testing.T, prog *Program, group OptimizerGroup) (TrainerHandle, []int64) {
	t.Helper()
	handles := make([]int64, 1)
	handle, err := FromData([]float32{1, -2, 3, -4}, 2, 2)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	handles[0] = handle
	trainer, err := CreateTrainer(prog, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{group},
		Weights: []WeightOptimizer{
			{GroupIndex: 0, Decay: true},
		},
		DefaultBaseLR: group.LR,
	})
	if err != nil {
		FreeHandles(handles)
		t.Fatalf("CreateTrainer: %v", err)
	}
	return trainer, handles
}

func cautiousWeightDecayInput() []TensorInput {
	return []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{2, 2}, Data: []float32{1, 1, -1, -1}},
	}
}

func readSingleWeight(t *testing.T, trainer TrainerHandle) []float32 {
	t.Helper()
	got := make([]float32, 4)
	if err := TrainerReadWeight(trainer, 0, got); err != nil {
		t.Fatalf("TrainerReadWeight: %v", err)
	}
	return got
}

func requireCloseSlice(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want %d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("elem %d got=%g want=%g diff=%g", i, got[i], want[i], diff)
		}
	}
}

func cautiousWeightDecayTestGroup(enabled bool, activationStep int) OptimizerGroup {
	return OptimizerGroup{
		Kind:                              OptimizerSGD,
		LR:                                0.1,
		Beta1:                             0,
		WeightDecay:                       0.2,
		CautiousWeightDecay:               enabled,
		CautiousWeightDecayActivationStep: activationStep,
	}
}

func TestTrainerCautiousWeightDecaySignMask(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := cautiousWeightDecayProgram(t)
	defer prog.Destroy()
	trainer, handles := cautiousWeightDecayTrainer(t, prog, cautiousWeightDecayTestGroup(true, 0))
	defer func() {
		TrainerDestroy(trainer)
		FreeHandles(handles)
	}()

	loss, err := TrainerStep(trainer, cautiousWeightDecayInput())
	if err != nil {
		t.Fatalf("TrainerStep: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss=%g, want finite", loss)
	}

	requireCloseSlice(t, readSingleWeight(t, trainer), []float32{0.955, -2.025, 3.025, -3.895}, 1e-5)
}

func TestTrainerCautiousWeightDecayActivationSchedule(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := cautiousWeightDecayProgram(t)
	defer prog.Destroy()

	standardTrainer, standardHandles := cautiousWeightDecayTrainer(t, prog, cautiousWeightDecayTestGroup(false, 0))
	defer func() {
		TrainerDestroy(standardTrainer)
		FreeHandles(standardHandles)
	}()
	scheduledTrainer, scheduledHandles := cautiousWeightDecayTrainer(t, prog, cautiousWeightDecayTestGroup(true, 1))
	defer func() {
		TrainerDestroy(scheduledTrainer)
		FreeHandles(scheduledHandles)
	}()

	if _, err := TrainerStep(standardTrainer, cautiousWeightDecayInput()); err != nil {
		t.Fatalf("TrainerStep standard first: %v", err)
	}
	if _, err := TrainerStep(scheduledTrainer, cautiousWeightDecayInput()); err != nil {
		t.Fatalf("TrainerStep scheduled first: %v", err)
	}
	requireCloseSlice(t, readSingleWeight(t, scheduledTrainer), readSingleWeight(t, standardTrainer), 1e-6)

	if _, err := TrainerStep(standardTrainer, cautiousWeightDecayInput()); err != nil {
		t.Fatalf("TrainerStep standard second: %v", err)
	}
	if _, err := TrainerStep(scheduledTrainer, cautiousWeightDecayInput()); err != nil {
		t.Fatalf("TrainerStep scheduled second: %v", err)
	}
	standard := readSingleWeight(t, standardTrainer)
	scheduled := readSingleWeight(t, scheduledTrainer)
	if math.Abs(float64(standard[1]-scheduled[1])) < 1e-4 || math.Abs(float64(standard[2]-scheduled[2])) < 1e-4 {
		t.Fatalf("scheduled CWD did not diverge from standard decay: standard=%v scheduled=%v", standard, scheduled)
	}
}
