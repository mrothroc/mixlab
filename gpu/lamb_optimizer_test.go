//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestLAMBOptimizerMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	tests := []struct {
		name        string
		w0          []float32
		w1          []float32
		driver0     []float32
		driver1     []float32
		decay0      bool
		decay1      bool
		weightDecay float64
		maxGradNorm float32
		steps       int
	}{
		{
			name:        "multi_weight_decay_and_no_decay",
			w0:          []float32{0.2, -0.4, 0.6, -0.8},
			w1:          []float32{0.5, -0.25, 0.125, -0.75},
			driver0:     []float32{0.3, -0.7, 1.1, -1.3},
			driver1:     []float32{-0.2, 0.4, -0.6, 0.8},
			decay0:      true,
			decay1:      false,
			weightDecay: 0.01,
			steps:       3,
		},
		{
			name:        "trust_ratio_fallback_and_clipping",
			w0:          []float32{0, 0, 0, 0},
			w1:          []float32{0.5, -0.25, 0.125, -0.75},
			driver0:     []float32{8, -4, 6, -2},
			driver1:     []float32{0, 0, 0, 0},
			decay0:      true,
			decay1:      false,
			weightDecay: 0.1,
			maxGradNorm: 0.75,
			steps:       2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			trainer, handles := newLAMBOracleTrainer(t, tt.w0, tt.w1, tt.decay0, tt.decay1, float32(tt.weightDecay), tt.maxGradNorm)
			defer TrainerDestroy(trainer)
			for _, h := range handles {
				defer FreeHandle(h)
			}

			state0 := lambOracleState{w: float64s(tt.w0), m: make([]float64, len(tt.w0)), v: make([]float64, len(tt.w0))}
			state1 := lambOracleState{w: float64s(tt.w1), m: make([]float64, len(tt.w1)), v: make([]float64, len(tt.w1))}
			grad0 := scaledGrad(tt.driver0)
			grad1 := scaledGrad(tt.driver1)

			inputs := []TensorInput{
				{Name: "driver0", DType: TensorFloat32, Shape: []int{2, 2}, Data: tt.driver0},
				{Name: "driver1", DType: TensorFloat32, Shape: []int{1, 4}, Data: tt.driver1},
			}
			for step := 1; step <= tt.steps; step++ {
				if _, err := TrainerStep(trainer, inputs); err != nil {
					t.Fatalf("TrainerStep step %d: %v", step, err)
				}
				g0 := append([]float64(nil), grad0...)
				g1 := append([]float64(nil), grad1...)
				clipOracleGrads(tt.maxGradNorm, g0, g1)
				lambOracleStep(&state0, g0, step, 0.05, 0.9, 0.999, 1e-6, tt.weightDecay, tt.decay0)
				lambOracleStep(&state1, g1, step, 0.05, 0.9, 0.999, 1e-6, tt.weightDecay, tt.decay1)
			}

			got0 := readTrainerWeight(t, trainer, 0)
			got1 := readTrainerWeight(t, trainer, 1)
			assertCloseSlice(t, got0, state0.w, 6e-5)
			assertCloseSlice(t, got1, state1.w, 6e-5)
		})
	}
}

func newLAMBOracleTrainer(t *testing.T, w0, w1 []float32, decay0, decay1 bool, weightDecay, maxGradNorm float32) (TrainerHandle, []int64) {
	t.Helper()
	prog := ir.NewProgram(2)
	prog.DeclareInput("driver0", ir.TensorFloat32, []int{2, 2})
	prog.DeclareInput("driver1", ir.TensorFloat32, []int{1, 4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Mul("w0", "driver0", "prod0")
	prog.MeanAxis("prod0", 0, "mean0")
	prog.MeanAxis("mean0", 0, "loss0")
	prog.Mul("w1", "driver1", "prod1")
	prog.MeanAxis("prod1", 0, "mean1")
	prog.MeanAxis("mean1", 0, "loss1")
	prog.Add("loss0", "loss1", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	t.Cleanup(gpuProg.Destroy)

	h0, err := FromDataShape(append([]float32(nil), w0...), []int{2, 2})
	if err != nil {
		t.Fatalf("FromDataShape w0: %v", err)
	}
	h1, err := FromDataShape(append([]float32(nil), w1...), []int{1, 4})
	if err != nil {
		FreeHandle(h0)
		t.Fatalf("FromDataShape w1: %v", err)
	}
	trainer, err := CreateTrainer(gpuProg, []int64{h0, h1}, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerLAMB,
			LR:          0.05,
			Beta1:       0.9,
			Beta2:       0.999,
			Epsilon:     1e-6,
			WeightDecay: weightDecay,
		}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0, Decay: decay0},
			{GroupIndex: 0, Decay: decay1},
		},
		MaxGradNorm:   maxGradNorm,
		DefaultBaseLR: 0.05,
	})
	if err != nil {
		FreeHandle(h0)
		FreeHandle(h1)
		t.Fatalf("CreateTrainer: %v", err)
	}
	return trainer, []int64{h0, h1}
}

type lambOracleState struct {
	w []float64
	m []float64
	v []float64
}

func lambOracleStep(state *lambOracleState, grad []float64, step int, lr, beta1, beta2, eps, weightDecay float64, decay bool) {
	update := make([]float64, len(state.w))
	b1t := 1 - math.Pow(beta1, float64(step))
	b2t := 1 - math.Pow(beta2, float64(step))
	for i := range state.w {
		state.m[i] = beta1*state.m[i] + (1-beta1)*grad[i]
		state.v[i] = beta2*state.v[i] + (1-beta2)*grad[i]*grad[i]
		mhat := state.m[i] / b1t
		vhat := state.v[i] / b2t
		update[i] = mhat / (math.Sqrt(vhat) + eps)
		if decay && weightDecay > 0 {
			update[i] += weightDecay * state.w[i]
		}
	}
	trustRatio := 1.0
	wNorm := l2Norm(state.w)
	updateNorm := l2Norm(update)
	if wNorm > 0 && updateNorm > 0 {
		ratio := wNorm / updateNorm
		if !math.IsNaN(ratio) && !math.IsInf(ratio, 0) {
			trustRatio = ratio
		}
	}
	for i := range state.w {
		state.w[i] -= lr * trustRatio * update[i]
	}
}

func scaledGrad(driver []float32) []float64 {
	out := make([]float64, len(driver))
	scale := float64(len(driver))
	for i, v := range driver {
		out[i] = float64(v) / scale
	}
	return out
}

func clipOracleGrads(maxGradNorm float32, grads ...[]float64) {
	if maxGradNorm <= 0 {
		return
	}
	normSq := 0.0
	for _, grad := range grads {
		for _, v := range grad {
			normSq += v * v
		}
	}
	norm := math.Sqrt(normSq)
	if norm == 0 || norm <= float64(maxGradNorm) {
		return
	}
	scale := float64(maxGradNorm) / (norm + 1e-6)
	for _, grad := range grads {
		for i := range grad {
			grad[i] *= scale
		}
	}
}

func l2Norm(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func float64s(values []float32) []float64 {
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = float64(v)
	}
	return out
}

func readTrainerWeight(t *testing.T, trainer TrainerHandle, idx int) []float32 {
	t.Helper()
	size, err := TrainerWeightSize(trainer, idx)
	if err != nil {
		t.Fatalf("TrainerWeightSize(%d): %v", idx, err)
	}
	out := make([]float32, size)
	if err := TrainerReadWeight(trainer, idx, out); err != nil {
		t.Fatalf("TrainerReadWeight(%d): %v", idx, err)
	}
	return out
}

func assertCloseSlice(t *testing.T, got []float32, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i]) - want[i]); diff > tol {
			t.Fatalf("value %d got=%0.8f want=%0.8f diff=%g tol=%g\ngot=%v\nwant=%v", i, got[i], want[i], diff, tol, got, want)
		}
	}
}
