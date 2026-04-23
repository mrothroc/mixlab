//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func makePipelineTestProgram() *ir.Program {
	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, 4})
	prog.DeclareInput("targets", ir.TensorInt32, []int{4})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{4, 2}, "emb_flat")
	prog.MatMul("emb_flat", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")
	return prog
}

func pipelineTestWeights() [][]float32 {
	return [][]float32{
		{
			0.1, 0.2,
			0.3, 0.4,
			0.5, 0.6,
			0.7, 0.8,
		},
		{
			0.2, -0.1, 0.0, 0.3,
			0.4, 0.1, -0.2, 0.5,
		},
	}
}

func pipelineTestSpec() TrainerOptimizerSpec {
	return TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:        OptimizerAdamW,
			LR:          0.001,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			WeightDecay: 0.01,
		}},
		Weights: []WeightOptimizer{
			{GroupIndex: 0, Decay: false},
			{GroupIndex: 0, Decay: true},
		},
		DefaultBaseLR: 0.001,
	}
}

func pipelineTestInputs(step int) []TensorInput {
	patterns := []struct {
		tokens  []int32
		targets []int32
	}{
		{tokens: []int32{0, 1, 2, 3}, targets: []int32{1, 2, 3, 0}},
		{tokens: []int32{3, 2, 1, 0}, targets: []int32{2, 1, 0, 3}},
		{tokens: []int32{1, 3, 0, 2}, targets: []int32{3, 0, 2, 1}},
	}
	choice := patterns[step%len(patterns)]
	return []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{1, 4}, Data: append([]int32(nil), choice.tokens...)},
		{Name: "targets", DType: TensorInt32, Shape: []int{4}, Data: append([]int32(nil), choice.targets...)},
	}
}

func createPipelineTestTrainer(t *testing.T, gpuProg *Program) TrainerHandle {
	t.Helper()

	weights := pipelineTestWeights()
	handles := make([]int64, len(weights))
	for i, data := range weights {
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
	trainer, err := CreateTrainer(gpuProg, handles, pipelineTestSpec())
	if err != nil {
		FreeHandles(handles)
		t.Fatalf("CreateTrainer: %v", err)
	}
	t.Cleanup(func() {
		_ = TrainerFlush(trainer)
		TrainerDestroy(trainer)
		FreeHandles(handles)
	})
	return trainer
}

func readAllTrainerWeights(t *testing.T, trainer TrainerHandle) [][]float32 {
	t.Helper()
	nWeights, err := TrainerNumWeights(trainer)
	if err != nil {
		t.Fatalf("TrainerNumWeights: %v", err)
	}
	out := make([][]float32, nWeights)
	for i := 0; i < nWeights; i++ {
		size, err := TrainerWeightSize(trainer, i)
		if err != nil {
			t.Fatalf("TrainerWeightSize(%d): %v", i, err)
		}
		data := make([]float32, size)
		if err := TrainerReadWeight(trainer, i, data); err != nil {
			t.Fatalf("TrainerReadWeight(%d): %v", i, err)
		}
		out[i] = data
	}
	return out
}

func TestTrainerPipelineCollectsSubmittedLossesInOrder(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	syncTrainer := createPipelineTestTrainer(t, gpuProg)
	pipelineTrainer := createPipelineTestTrainer(t, gpuProg)

	input0 := pipelineTestInputs(0)
	input1 := pipelineTestInputs(1)

	wantLoss0, err := TrainerStep(syncTrainer, input0)
	if err != nil {
		t.Fatalf("TrainerStep sync step 0: %v", err)
	}
	wantLoss1, err := TrainerStep(syncTrainer, input1)
	if err != nil {
		t.Fatalf("TrainerStep sync step 1: %v", err)
	}

	if err := TrainerSubmitStep(pipelineTrainer, input0); err != nil {
		t.Fatalf("TrainerSubmitStep step 0: %v", err)
	}
	if err := TrainerSubmitStep(pipelineTrainer, input1); err != nil {
		t.Fatalf("TrainerSubmitStep step 1: %v", err)
	}
	gotLoss0, err := TrainerCollectLoss(pipelineTrainer)
	if err != nil {
		t.Fatalf("TrainerCollectLoss step 0: %v", err)
	}
	gotLoss1, err := TrainerCollectLoss(pipelineTrainer)
	if err != nil {
		t.Fatalf("TrainerCollectLoss step 1: %v", err)
	}

	if diff := math.Abs(float64(gotLoss0 - wantLoss0)); diff > 1e-6 {
		t.Fatalf("step 0 loss mismatch: got=%g want=%g diff=%g", gotLoss0, wantLoss0, diff)
	}
	if diff := math.Abs(float64(gotLoss1 - wantLoss1)); diff > 1e-6 {
		t.Fatalf("step 1 loss mismatch: got=%g want=%g diff=%g", gotLoss1, wantLoss1, diff)
	}

	wantWeights := readAllTrainerWeights(t, syncTrainer)
	gotWeights := readAllTrainerWeights(t, pipelineTrainer)
	for i := range wantWeights {
		for j := range wantWeights[i] {
			if diff := math.Abs(float64(gotWeights[i][j] - wantWeights[i][j])); diff > 1e-6 {
				t.Fatalf("weight[%d][%d] mismatch: got=%g want=%g diff=%g", i, j, gotWeights[i][j], wantWeights[i][j], diff)
			}
		}
	}
}

func TestTrainerPipelineMatchesSynchronousTrainingExactly(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(makePipelineTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	syncTrainer := createPipelineTestTrainer(t, gpuProg)
	pipelineTrainer := createPipelineTestTrainer(t, gpuProg)

	const steps = 10
	wantLosses := make([]float32, steps)
	for step := 0; step < steps; step++ {
		loss, err := TrainerStep(syncTrainer, pipelineTestInputs(step))
		if err != nil {
			t.Fatalf("TrainerStep sync step %d: %v", step, err)
		}
		wantLosses[step] = loss
	}

	if err := TrainerSubmitStep(pipelineTrainer, pipelineTestInputs(0)); err != nil {
		t.Fatalf("TrainerSubmitStep initial: %v", err)
	}
	gotLosses := make([]float32, 0, steps)
	for step := 0; step < steps; step++ {
		if step < steps-1 {
			if err := TrainerSubmitStep(pipelineTrainer, pipelineTestInputs(step+1)); err != nil {
				t.Fatalf("TrainerSubmitStep step %d: %v", step+1, err)
			}
		}
		loss, err := TrainerCollectLoss(pipelineTrainer)
		if err != nil {
			t.Fatalf("TrainerCollectLoss step %d: %v", step, err)
		}
		gotLosses = append(gotLosses, loss)
	}

	for step := 0; step < steps; step++ {
		if diff := math.Abs(float64(gotLosses[step] - wantLosses[step])); diff > 1e-6 {
			t.Fatalf("loss[%d] mismatch: got=%g want=%g diff=%g", step, gotLosses[step], wantLosses[step], diff)
		}
	}

	wantWeights := readAllTrainerWeights(t, syncTrainer)
	gotWeights := readAllTrainerWeights(t, pipelineTrainer)
	for i := range wantWeights {
		for j := range wantWeights[i] {
			if diff := math.Abs(float64(gotWeights[i][j] - wantWeights[i][j])); diff > 1e-6 {
				t.Fatalf("weight[%d][%d] mismatch: got=%g want=%g diff=%g", i, j, gotWeights[i][j], wantWeights[i][j], diff)
			}
		}
	}
}
