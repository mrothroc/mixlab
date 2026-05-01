//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTrainerNorMuonChangesMatrixUpdate(t *testing.T) {
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

	makeTrainer := func(normalization MuonNormalization) (TrainerHandle, []int64) {
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
					Kind:              OptimizerMuon,
					LR:                lr,
					Beta1:             0.9,
					Beta2:             0.95,
					Epsilon:           1e-8,
					BackendSteps:      5,
					Nesterov:          true,
					MuonNormalization: normalization,
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

	baseTrainer, baseHandles := makeTrainer(MuonNormalizationNone)
	defer func() {
		TrainerDestroy(baseTrainer)
		FreeHandles(baseHandles)
	}()
	norTrainer, norHandles := makeTrainer(MuonNormalizationNorMuon)
	defer func() {
		TrainerDestroy(norTrainer)
		FreeHandles(norHandles)
	}()

	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: []int32{1, 2, 3, 0}},
	}
	baseLoss, err := TrainerStep(baseTrainer, inputs)
	if err != nil {
		t.Fatalf("TrainerStep(base): %v", err)
	}
	norLoss, err := TrainerStep(norTrainer, inputs)
	if err != nil {
		t.Fatalf("TrainerStep(normuon): %v", err)
	}
	if !isFinitePositive(baseLoss) || !isFinitePositive(norLoss) {
		t.Fatalf("non-finite loss: base=%g normuon=%g", baseLoss, norLoss)
	}

	baseW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(baseTrainer, 1, baseW1); err != nil {
		t.Fatalf("TrainerReadWeight(base): %v", err)
	}
	norW1 := make([]float32, len(w1))
	if err := TrainerReadWeight(norTrainer, 1, norW1); err != nil {
		t.Fatalf("TrainerReadWeight(normuon): %v", err)
	}

	maxDiff := 0.0
	baseUpdateNorm := 0.0
	norUpdateNorm := 0.0
	for i := range w1 {
		maxDiff = math.Max(maxDiff, math.Abs(float64(baseW1[i]-norW1[i])))
		baseUpdate := float64(w1[i]-baseW1[i]) / float64(lr)
		norUpdate := float64(w1[i]-norW1[i]) / float64(lr)
		baseUpdateNorm += baseUpdate * baseUpdate
		norUpdateNorm += norUpdate * norUpdate
	}
	if maxDiff < 1e-7 {
		t.Fatalf("NorMuon did not change matrix update; max diff=%g", maxDiff)
	}
	if diff := math.Abs(math.Sqrt(baseUpdateNorm) - math.Sqrt(norUpdateNorm)); diff > 0.12 {
		t.Fatalf("NorMuon update norm diff=%g want <=0.12", diff)
	}
}
