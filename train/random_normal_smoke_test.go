//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestRandomNormalForwardBackwardSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	const (
		modelDim = 8
		seqLen   = 4
		vocab    = modelDim
	)

	cfg := &ArchConfig{
		Name:          "random_normal_smoke",
		ModelDim:      modelDim,
		VocabSize:     vocab,
		SeqLen:        seqLen,
		TieEmbeddings: true,
		Training: TrainingSpec{
			Steps:       1,
			LR:          1e-2,
			EmbedLR:     1e-2,
			ScalarLR:    1e-2,
			Beta1:       0.9,
			Beta2:       0.95,
			Epsilon:     1e-8,
			Seed:        7,
			BatchTokens: seqLen,
			GradClip:    0,
			WeightDecay: 0,
		},
	}

	prog := buildRandomNormalSmokeProgram(modelDim, seqLen, vocab)
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	if err := trainer.SetWeightGPU("embed", randomNormalSmokeEmbedWeights(vocab, modelDim)); err != nil {
		t.Fatalf("SetWeightGPU(embed): %v", err)
	}
	if err := trainer.SetWeightGPU("final_norm", randomNormalSmokeScaleWeights(modelDim)); err != nil {
		t.Fatalf("SetWeightGPU(final_norm): %v", err)
	}

	before, err := trainer.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights(before): %v", err)
	}

	tokens := []int{0, 1, 2, 3}
	targets := []int{1, 2, 3, 4}
	loss, err := trainer.TrainStepGPU(tokens, targets, 1, seqLen, float32(cfg.Training.LR))
	if err != nil {
		t.Fatalf("TrainStepGPU: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss: %g", loss)
	}

	if err := trainer.FlushGPU(); err != nil {
		t.Fatalf("FlushGPU: %v", err)
	}

	embedIdx := randomNormalWeightIndex(t, trainer.shapes, "embed")
	scaleIdx := randomNormalWeightIndex(t, trainer.shapes, "final_norm")

	inputs, err := trainer.makeInputs(tokens, targets, 1, seqLen)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}
	gradSignal, err := gpu.TrainerComputeMeanSquareGrads(trainer.handle, inputs, "logits")
	if err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads(logits): %v", err)
	}
	if math.IsNaN(float64(gradSignal)) || math.IsInf(float64(gradSignal), 0) {
		t.Fatalf("non-finite grad signal: %g", gradSignal)
	}

	embedGrad := make([]float32, len(before[embedIdx]))
	if err := gpu.TrainerReadGrad(trainer.handle, embedIdx, embedGrad); err != nil {
		t.Fatalf("TrainerReadGrad(embed): %v", err)
	}
	scaleGrad := make([]float32, len(before[scaleIdx]))
	if err := gpu.TrainerReadGrad(trainer.handle, scaleIdx, scaleGrad); err != nil {
		t.Fatalf("TrainerReadGrad(final_norm): %v", err)
	}

	if max := randomNormalMaxAbs(embedGrad); max <= 1e-8 {
		t.Fatalf("embed gradient too small: max abs=%g", max)
	}
	if max := randomNormalMaxAbs(scaleGrad); max <= 1e-8 {
		t.Fatalf("final_norm gradient too small: max abs=%g", max)
	}

	after, err := trainer.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights(after): %v", err)
	}

	if diff := randomNormalMaxDiff(before[embedIdx], after[embedIdx]); diff <= 1e-8 {
		t.Fatalf("embed did not update: max diff=%g", diff)
	}
	if diff := randomNormalMaxDiff(before[scaleIdx], after[scaleIdx]); diff <= 1e-8 {
		t.Fatalf("final_norm did not update: max diff=%g", diff)
	}
	if !randomNormalAllFinite(after[embedIdx]) || !randomNormalAllFinite(after[scaleIdx]) {
		t.Fatal("updated weights contain NaN or Inf")
	}

	t.Logf(
		"loss=%0.6f grad_signal=%0.6f embed_grad_max=%0.6g final_norm_grad_max=%0.6g embed_update_max=%0.6g final_norm_update_max=%0.6g",
		loss,
		gradSignal,
		randomNormalMaxAbs(embedGrad),
		randomNormalMaxAbs(scaleGrad),
		randomNormalMaxDiff(before[embedIdx], after[embedIdx]),
		randomNormalMaxDiff(before[scaleIdx], after[scaleIdx]),
	)
}

func buildRandomNormalSmokeProgram(modelDim, seqLen, vocab int) *ir.Program {
	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{1, seqLen})
	prog.DeclareInput("targets", ir.TensorInt32, []int{seqLen})
	prog.DeclareOutput("logits", ir.TensorFloat32, []int{seqLen, vocab})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{seqLen, modelDim}, "emb_flat")
	prog.RandomNormal([]int{seqLen, modelDim}, 0.0, 1.0, "noise")
	prog.Add("emb_flat", "noise", "noisy")
	prog.Mul("noisy", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")
	return prog
}

func randomNormalSmokeEmbedWeights(vocab, modelDim int) []float32 {
	data := make([]float32, vocab*modelDim)
	for row := 0; row < vocab; row++ {
		for col := 0; col < modelDim; col++ {
			data[row*modelDim+col] = 0.05 * float32((row+1)*(col+2)) / float32(modelDim)
		}
	}
	return data
}

func randomNormalSmokeScaleWeights(modelDim int) []float32 {
	data := make([]float32, modelDim)
	for i := range data {
		data[i] = 0.4 + 0.05*float32(i)
	}
	return data
}

func randomNormalWeightIndex(t *testing.T, shapes []WeightShape, name string) int {
	t.Helper()
	for i, shape := range shapes {
		if shape.Name == name {
			return i
		}
	}
	t.Fatalf("missing weight %q", name)
	return -1
}

func randomNormalMaxAbs(data []float32) float32 {
	maxAbs := float32(0)
	for _, v := range data {
		abs := float32(math.Abs(float64(v)))
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs
}

func randomNormalMaxDiff(a, b []float32) float32 {
	maxDiff := float32(0)
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func randomNormalAllFinite(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}
