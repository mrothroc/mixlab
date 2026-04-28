//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func readWeightGrad(t *testing.T, trainer TrainerHandle, weightIdx, size int) []float32 {
	t.Helper()

	grad := make([]float32, size)
	if err := TrainerReadGrad(trainer, weightIdx, grad); err != nil {
		t.Fatalf("TrainerReadGrad(%d): %v", weightIdx, err)
	}
	return grad
}

func requireZeroGrad(t *testing.T, grad []float32) {
	t.Helper()

	for i, v := range grad {
		if math.Abs(float64(v)) > 1e-7 {
			t.Fatalf("grad[%d]=%g, want 0", i, v)
		}
	}
}

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

	requireZeroGrad(t, readWeightGrad(t, trainer, 0, 6))
}

func TestStopGradientBlocksMatMulWeightGradient(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(2)
	prog.DeclareInput("dummy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("logits", ir.TensorFloat32, []int{2, 2})
	prog.StopGradient("w0", "blocked")
	prog.MatMul("blocked", "w1", "logits")

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

	w1, err := FromData([]float32{0.25, -0.5, 1.5, 0.75, -1.25, 2.0}, 3, 2)
	if err != nil {
		t.Fatalf("FromData w1: %v", err)
	}
	defer FreeHandle(w1)

	trainer, err := CreateTrainer(gpuProg, []int64{w0, w1}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:    OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.95,
			Epsilon: 1e-8,
		}},
		Weights:       []WeightOptimizer{{GroupIndex: 0}, {GroupIndex: 0}},
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
	}}, "logits"); err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(logits): %v", err)
	}

	requireZeroGrad(t, readWeightGrad(t, trainer, 0, 6))
}

func TestStopGradientBlocksScatterPositionsWeightGradient(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		B = 1
		T = 4
		K = 2
		D = 3
	)

	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B, T, D})
	prog.DeclareInput("positions", ir.TensorInt32, []int{K})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{B, T, D})
	prog.Reshape("w0", []int{B, K, D}, "source")
	prog.StopGradient("source", "blocked")
	prog.ScatterPositions("x", "blocked", "positions", "out", B, T, K, D)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData([]float32{
		1, -2, 3,
		-4, 5, -6,
	}, K, D)
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

	if _, err := TrainerComputeMeanSquareGrads(trainer, []TensorInput{
		{
			Name:  "x",
			DType: TensorFloat32,
			Shape: []int{B, T, D},
			Data: []float32{
				0.1, 0.2, 0.3,
				0.4, 0.5, 0.6,
				0.7, 0.8, 0.9,
				1.0, 1.1, 1.2,
			},
		},
		{
			Name:  "positions",
			DType: TensorInt32,
			Shape: []int{K},
			Data:  []int32{1, 3},
		},
	}, "out"); err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(out): %v", err)
	}

	requireZeroGrad(t, readWeightGrad(t, trainer, 0, K*D))
}

func TestStopGradientBlocksGatherPositionsWeightGradient(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		B = 1
		T = 4
		K = 2
		D = 3
	)

	prog := ir.NewProgram(1)
	prog.DeclareInput("positions", ir.TensorInt32, []int{K})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{B, K, D})
	prog.Reshape("w0", []int{B, T, D}, "source")
	prog.StopGradient("source", "blocked")
	prog.GatherPositions("blocked", "positions", "out", B, K, D)

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData([]float32{
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
		1.0, 1.1, 1.2,
	}, B*T, D)
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
		Name:  "positions",
		DType: TensorInt32,
		Shape: []int{K},
		Data:  []int32{0, 2},
	}}, "out"); err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(out): %v", err)
	}

	requireZeroGrad(t, readWeightGrad(t, trainer, 0, B*T*D))
}
