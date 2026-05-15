//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"runtime"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTrainerMuonEqRRowNormalizesMatrixUpdate(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareInput("targets", ir.TensorInt32, []int{4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{4, 3}, "emb_flat")
	prog.MatMul("emb_flat", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0 := []float32{
		0.11, 0.27, -0.15,
		0.33, 0.49, 0.05,
		0.52, -0.68, 0.17,
		0.71, 0.89, -0.23,
	}
	w1 := []float32{
		0.203, -0.117, 0.041, 0.287,
		0.411, 0.096, -0.183, 0.534,
		-0.151, 0.222, 0.317, -0.409,
	}
	const lr = float32(0.01)

	makeTrainer := func(rowNormalize bool) (TrainerHandle, []int64) {
		t.Helper()
		handles := make([]int64, 2)
		for i, data := range [][]float32{w0, w1} {
			rows, cols := 4, 3
			if i == 1 {
				rows, cols = 3, 4
			}
			handle, err := FromData(append([]float32(nil), data...), rows, cols)
			if err != nil {
				t.Fatalf("FromData(%d): %v", i, err)
			}
			handles[i] = handle
		}
		trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
			Groups: []OptimizerGroup{
				{
					Kind:        OptimizerAdamW,
					LR:          0,
					Beta1:       0.9,
					Beta2:       0.95,
					Epsilon:     1e-8,
					WeightDecay: 0,
				},
				{
					Kind:         OptimizerMuon,
					LR:           lr,
					Beta1:        0.9,
					Beta2:        0.95,
					Epsilon:      1e-8,
					BackendSteps: 5,
					Nesterov:     true,
					RowNormalize: rowNormalize,
				},
			},
			Weights: []WeightOptimizer{
				{GroupIndex: 0, Decay: false},
				{GroupIndex: 1, Decay: false},
			},
			DefaultBaseLR: lr,
		})
		if err != nil {
			FreeHandles(handles)
			t.Fatalf("CreateTrainer: %v", err)
		}
		return trainer, handles
	}

	baseTrainer, baseHandles := makeTrainer(false)
	defer func() {
		TrainerDestroy(baseTrainer)
		FreeHandles(baseHandles)
	}()
	eqTrainer, eqHandles := makeTrainer(true)
	defer func() {
		TrainerDestroy(eqTrainer)
		FreeHandles(eqHandles)
	}()

	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: []int32{1, 2, 3, 0}},
	}
	for step := 0; step < 1; step++ {
		baseLoss, err := TrainerStep(baseTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(base, step=%d): %v", step, err)
		}
		eqLoss, err := TrainerStep(eqTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(eq_r, step=%d): %v", step, err)
		}
		if !isFinitePositive(baseLoss) || !isFinitePositive(eqLoss) {
			t.Fatalf("non-finite loss at step %d: base=%g eq_r=%g", step, baseLoss, eqLoss)
		}
	}

	baseW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(baseTrainer, 1, baseW1); err != nil {
		t.Fatalf("TrainerReadWeight(base): %v", err)
	}
	eqW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(eqTrainer, 1, eqW1); err != nil {
		t.Fatalf("TrainerReadWeight(eq_r): %v", err)
	}

	maxDiff := 0.0
	for i := range baseW1 {
		maxDiff = math.Max(maxDiff, math.Abs(float64(baseW1[i]-eqW1[i])))
	}
	if maxDiff < 1e-7 {
		t.Fatalf("MuonEq-R did not change matrix update; max diff=%g", maxDiff)
	}

	for row := 0; row < 3; row++ {
		sumSq := 0.0
		for col := 0; col < 4; col++ {
			idx := row*4 + col
			update := float64(w1[idx]-eqW1[idx]) / float64(lr)
			sumSq += update * update
		}
		if diff := math.Abs(math.Sqrt(sumSq) - 1.0); diff > 0.08 {
			t.Fatalf("row %d normalized update norm diff=%g want <=0.08", row, diff)
		}
	}

	for step := 1; step < 10; step++ {
		baseLoss, err := TrainerStep(baseTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(base, step=%d): %v", step, err)
		}
		eqLoss, err := TrainerStep(eqTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(eq_r, step=%d): %v", step, err)
		}
		if !isFinitePositive(baseLoss) || !isFinitePositive(eqLoss) {
			t.Fatalf("non-finite loss at step %d: base=%g eq_r=%g", step, baseLoss, eqLoss)
		}
	}
	if err := TrainerReadWeight(baseTrainer, 1, baseW1); err != nil {
		t.Fatalf("TrainerReadWeight(base after 10 steps): %v", err)
	}
	if err := TrainerReadWeight(eqTrainer, 1, eqW1); err != nil {
		t.Fatalf("TrainerReadWeight(eq_r after 10 steps): %v", err)
	}
	maxDiff = 0.0
	for i := range baseW1 {
		maxDiff = math.Max(maxDiff, math.Abs(float64(baseW1[i]-eqW1[i])))
	}
	if maxDiff < 1e-6 {
		t.Fatalf("MuonEq-R stayed too close to Muon after 10 steps; max diff=%g", maxDiff)
	}
}

func isFinitePositive(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) && v > 0
}
