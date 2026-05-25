//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedCrossEntropyMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		nTokens = 4
		vocab   = 3
		tol     = 1e-5
	)
	logits := []float32{
		2.0, 0.5, -1.0,
		0.1, 1.7, 0.2,
		-0.3, 0.4, 2.2,
		1.0, -0.5, 0.25,
	}
	targets := []int32{0, 1, 2, 1}
	mask := []float32{1, 0, 1, 0}

	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{nTokens, vocab})
	prog.DeclareInput("targets", ir.TensorInt32, []int{nTokens})
	prog.DeclareInput("loss_mask", ir.TensorFloat32, []int{nTokens})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("per_token_nll", ir.TensorFloat32, []int{nTokens})
	prog.MaskedCrossEntropy("logits", "targets", "loss_mask", "loss")
	prog.MaskedCrossEntropyPerToken("logits", "targets", "loss_mask", "per_token_nll")

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

	trainer, err := CreateTrainer(gpuProg, []int64{dummyWeight}, TrainerOptimizerSpec{
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

	inputs := []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{nTokens, vocab}, Data: logits},
		{Name: "targets", DType: TensorInt32, Shape: []int{nTokens}, Data: targets},
		{Name: "loss_mask", DType: TensorFloat32, Shape: []int{nTokens}, Data: mask},
	}

	gotLoss, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}
	wantLoss, wantPerToken := maskedCEOracle(logits, targets, mask, vocab)
	if diff := math.Abs(float64(gotLoss - wantLoss)); diff > tol {
		t.Fatalf("loss = %g, want %g, diff %g", gotLoss, wantLoss, diff)
	}

	gotPerToken, err := TrainerEvaluatePerToken(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluatePerToken: %v", err)
	}
	if len(gotPerToken) != nTokens {
		t.Fatalf("len(per_token_nll) = %d, want %d", len(gotPerToken), nTokens)
	}
	for i := range gotPerToken {
		if diff := math.Abs(float64(gotPerToken[i] - wantPerToken[i])); diff > tol {
			t.Fatalf("per_token[%d] = %g, want %g, diff %g", i, gotPerToken[i], wantPerToken[i], diff)
		}
	}
}

func maskedCEOracle(logits []float32, targets []int32, mask []float32, vocab int) (float32, []float32) {
	perToken := make([]float32, len(targets))
	sum := 0.0
	count := 0.0
	for i, target := range targets {
		row := logits[i*vocab : (i+1)*vocab]
		maxLogit := float64(row[0])
		for _, v := range row[1:] {
			if float64(v) > maxLogit {
				maxLogit = float64(v)
			}
		}
		expSum := 0.0
		for _, v := range row {
			expSum += math.Exp(float64(v) - maxLogit)
		}
		nll := float32(-(float64(row[target]) - maxLogit - math.Log(expSum)))
		if mask[i] > 0 {
			perToken[i] = nll
			sum += float64(nll)
			count++
		}
	}
	if count == 0 {
		return 0, perToken
	}
	return float32(sum / count), perToken
}
