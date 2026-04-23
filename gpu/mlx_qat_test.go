//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTrainerQATChangesMatrixWeightUpdate(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareInput("targets", ir.TensorInt32, []int{1, 4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{4, 2}, "emb_flat")
	prog.MatMul("emb_flat", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0 := []float32{
		0.11, 0.27,
		0.33, 0.49,
		0.52, 0.68,
		0.71, 0.89,
	}
	w1 := []float32{
		0.203, -0.117, 0.041, 0.287,
		0.411, 0.096, -0.183, 0.534,
	}

	makeTrainer := func() (TrainerHandle, []int64) {
		t.Helper()
		handles := make([]int64, 2)
		for i, data := range [][]float32{w0, w1} {
			rows, cols := 4, 2
			if i == 1 {
				rows, cols = 2, 4
			}
			handle, err := FromData(append([]float32(nil), data...), rows, cols)
			if err != nil {
				t.Fatalf("FromData(%d): %v", i, err)
			}
			handles[i] = handle
		}
		trainer, err := CreateTrainer(gpuProg, handles, TrainerOptimizerSpec{
			Groups: []OptimizerGroup{{
				Kind:        OptimizerAdamW,
				LR:          0.001,
				Beta1:       0.9,
				Beta2:       0.95,
				Epsilon:     1e-8,
				WeightDecay: 0.0,
			}},
			Weights: []WeightOptimizer{
				{GroupIndex: 0, Decay: false},
				{GroupIndex: 0, Decay: false},
			},
			DefaultBaseLR: 0.001,
		})
		if err != nil {
			FreeHandles(handles)
			t.Fatalf("CreateTrainer: %v", err)
		}
		return trainer, handles
	}

	baseTrainer, baseHandles := makeTrainer()
	defer func() {
		TrainerDestroy(baseTrainer)
		FreeHandles(baseHandles)
	}()

	qatTrainer, qatHandles := makeTrainer()
	defer func() {
		TrainerDestroy(qatTrainer)
		FreeHandles(qatHandles)
	}()
	if err := TrainerSetQAT(qatTrainer, "int6"); err != nil {
		t.Fatalf("TrainerSetQAT: %v", err)
	}

	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: []int32{1, 2, 3, 0}},
	}

	maxLossDiff := 0.0
	for step := 0; step < 2; step++ {
		baseLoss, err := TrainerStep(baseTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(base, step=%d): %v", step, err)
		}
		qatLoss, err := TrainerStep(qatTrainer, inputs)
		if err != nil {
			t.Fatalf("TrainerStep(qat, step=%d): %v", step, err)
		}
		maxLossDiff = math.Max(maxLossDiff, math.Abs(float64(baseLoss-qatLoss)))
	}
	if maxLossDiff < 1e-7 {
		t.Fatalf("losses unexpectedly matched across steps; max diff=%g", maxLossDiff)
	}

	baseW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(baseTrainer, 1, baseW1); err != nil {
		t.Fatalf("TrainerReadWeight(base): %v", err)
	}
	qatW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(qatTrainer, 1, qatW1); err != nil {
		t.Fatalf("TrainerReadWeight(qat): %v", err)
	}

	maxDiff := 0.0
	for i := range baseW1 {
		maxDiff = math.Max(maxDiff, math.Abs(float64(baseW1[i]-qatW1[i])))
	}
	if maxDiff < 1e-7 {
		t.Fatalf("QAT did not change matrix weight update; max diff=%g", maxDiff)
	}
}
