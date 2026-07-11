//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedSymmetricKLMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		seqLen = 3
		vocab  = 3
	)
	logits := []float32{
		0.2, 1.1, -0.4, 1.4, -0.1, 0.3, 0.7, -0.8, 0.9,
		-0.3, 0.5, 1.2, 0.6, 1.0, -0.2, 1.1, -0.4, 0.4,
		// Second pair is inactive and must not affect the result.
		4, -3, 2, -5, 3, 1, 7, 2, -4,
		-2, 6, 1, 3, -7, 2, 8, -1, 0,
	}
	mask := []float32{
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
		0, 0, 0,
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{len(mask), vocab})
	prog.DeclareInput("invariance_loss_mask", ir.TensorFloat32, []int{len(mask)})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.MaskedSymmetricKL("logits", "invariance_loss_mask", seqLen, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	weight, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(weight)
	trainer, err := CreateTrainer(gpuProg, []int64{weight}, TrainerOptimizerSpec{
		Groups:  []OptimizerGroup{{Kind: OptimizerAdamW, LR: 1e-3, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8}},
		Weights: []WeightOptimizer{{GroupIndex: 0}}, DefaultBaseLR: 1e-3,
	})
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)
	inputs := []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{len(mask), vocab}, Data: logits},
		{Name: "invariance_loss_mask", DType: TensorFloat32, Shape: []int{len(mask)}, Data: mask},
	}
	got, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}
	want := symmetricKLOracle(logits, mask, seqLen, vocab)
	if diff := math.Abs(float64(got - want)); diff > 1e-5 {
		t.Fatalf("loss=%g want=%g diff=%g", got, want, diff)
	}

	poisoned := append([]float32(nil), logits...)
	for i := 2 * seqLen * vocab; i < len(poisoned); i++ {
		poisoned[i] *= -9
	}
	gotPoisoned, err := TrainerEvaluate(trainer, []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{len(mask), vocab}, Data: poisoned},
		{Name: "invariance_loss_mask", DType: TensorFloat32, Shape: []int{len(mask)}, Data: mask},
	})
	if err != nil {
		t.Fatalf("TrainerEvaluate poisoned: %v", err)
	}
	if diff := math.Abs(float64(gotPoisoned - got)); diff > 1e-6 {
		t.Fatalf("inactive pair changed loss: got=%g poisoned=%g", got, gotPoisoned)
	}
}

func TestMaskedSymmetricKLNearDegenerateDistributionHasFiniteLossAndGradients(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		seqLen = 1
		vocab  = 4
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{2})
	prog.DeclareInput("invariance_loss_mask", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "logits")
	prog.MaskedSymmetricKL("logits", "invariance_loss_mask", seqLen, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	// One almost one-hot view and one uniform view. This drove an unbounded
	// cross-distribution term before the probability-floor implementation.
	weights := []float32{
		80, -80, -80, -80,
		0, 0, 0, 0,
	}
	weight, err := FromDataShape(weights, []int{2, vocab})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)
	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{2}, Data: []int32{0, 1}},
		{Name: "invariance_loss_mask", DType: TensorFloat32, Shape: []int{2}, Data: []float32{1, 1}},
	}
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, []int64{weight}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	if !(loss >= 0) || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss=%g, want finite non-negative", loss)
	}
	if loss > 20 {
		t.Fatalf("loss=%g, want bounded collapsed-tail KL", loss)
	}
	if len(grads) != 1 || len(grads[0]) != len(weights) {
		t.Fatalf("gradient shape=%v, want one [%d] tensor", grads, len(weights))
	}
	for i, grad := range grads[0] {
		if math.IsNaN(float64(grad)) || math.IsInf(float64(grad), 0) {
			t.Fatalf("gradient[%d]=%g, want finite", i, grad)
		}
	}
}

func symmetricKLOracle(logits, mask []float32, seqLen, vocab int) float32 {
	pairs := len(mask) / (2 * seqLen)
	total := 0.0
	active := 0
	for pair := 0; pair < pairs; pair++ {
		base := pair * 2 * seqLen
		posA, posB := -1, -1
		for pos := 0; pos < seqLen; pos++ {
			if mask[base+pos] > 0 {
				posA = pos
			}
			if mask[base+seqLen+pos] > 0 {
				posB = pos
			}
		}
		if posA < 0 || posB < 0 {
			continue
		}
		pa := softmaxOracle(logits[(base+posA)*vocab : (base+posA+1)*vocab])
		pb := softmaxOracle(logits[(base+seqLen+posB)*vocab : (base+seqLen+posB+1)*vocab])
		klAB, klBA := 0.0, 0.0
		for i := 0; i < vocab; i++ {
			klAB += pa[i] * (math.Log(pa[i]) - math.Log(pb[i]))
			klBA += pb[i] * (math.Log(pb[i]) - math.Log(pa[i]))
		}
		total += 0.5 * (klAB + klBA)
		active++
	}
	if active == 0 {
		return 0
	}
	return float32(total / float64(active))
}

func softmaxOracle(row []float32) []float64 {
	maxValue := float64(row[0])
	for _, value := range row[1:] {
		if float64(value) > maxValue {
			maxValue = float64(value)
		}
	}
	denom := 0.0
	out := make([]float64, len(row))
	for i, value := range row {
		out[i] = math.Exp(float64(value) - maxValue)
		denom += out[i]
	}
	for i := range out {
		out[i] /= denom
	}
	return out
}
