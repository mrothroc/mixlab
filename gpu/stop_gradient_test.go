//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestStopGradientForwardIdentity(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2, 3})
	prog.DeclareOutput("y", ir.TensorFloat32, []int{2, 3})
	prog.StopGradient("x", "y")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	dummyWeight, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData dummy weight: %v", err)
	}
	defer FreeHandle(dummyWeight)

	x := []float32{1, -2, 3.5, -4.5, 5.25, -6.75}
	got, err := evalProgramOutput(gpuProg, []int64{dummyWeight}, []TensorInput{{
		Name:  "x",
		DType: TensorFloat32,
		Shape: []int{2, 3},
		Data:  x,
	}}, "y")
	if err != nil {
		t.Fatalf("evalProgramOutput(y): %v", err)
	}

	if len(got) != len(x) {
		t.Fatalf("len(y)=%d, want %d", len(got), len(x))
	}
	for i := range x {
		if diff := math.Abs(float64(got[i] - x[i])); diff > 1e-6 {
			t.Fatalf("y[%d]=%g, want %g (diff=%g)", i, got[i], x[i], diff)
		}
	}
}

func TestStopGradientZeroesWeightGradient(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("dummy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("blocked", ir.TensorFloat32, []int{2, 3})
	prog.StopGradient("w0", "blocked")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData([]float32{1, -2, 3, -4, 5, -6}, 2, 3)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	trainer, err := CreateTrainer(gpuProg, []int64{w0}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:    OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.95,
			Epsilon: 1e-8,
		}},
		Weights:       []WeightOptimizer{{GroupIndex: 0}},
		DefaultBaseLR: 1e-3,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	if _, err := TrainerComputeMeanSquareGrads(trainer, []TensorInput{{
		Name:  "dummy",
		DType: TensorFloat32,
		Shape: []int{1},
		Data:  []float32{0},
	}}, "blocked"); err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(blocked): %v", err)
	}

	grad := make([]float32, 6)
	if err := TrainerReadGrad(trainer, 0, grad); err != nil {
		t.Fatalf("TrainerReadGrad(0): %v", err)
	}
	for i, v := range grad {
		if math.Abs(float64(v)) > 1e-7 {
			t.Fatalf("grad[%d]=%g, want 0", i, v)
		}
	}
}
