//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestCrossEntropyPerTokenMatchesMean(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		nTokens = 3
		vocab   = 4
		tol     = 1e-5
	)

	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{nTokens, vocab})
	prog.DeclareInput("targets", ir.TensorInt32, []int{nTokens})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("per_token_nll", ir.TensorFloat32, []int{nTokens})
	prog.CrossEntropy("logits", "targets", "loss")
	prog.CrossEntropyPerToken("logits", "targets", "per_token_nll")

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
		{
			Name:  "logits",
			DType: TensorFloat32,
			Shape: []int{nTokens, vocab},
			Data: []float32{
				2.0, 0.5, -1.0, 0.0,
				0.1, 1.7, 0.2, -0.5,
				-0.3, 0.4, 2.2, 0.0,
			},
		},
		{
			Name:  "targets",
			DType: TensorInt32,
			Shape: []int{nTokens},
			Data:  []int32{0, 1, 2},
		},
	}

	meanLoss, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}

	perToken, err := TrainerEvaluatePerToken(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluatePerToken: %v", err)
	}
	if len(perToken) != nTokens {
		t.Fatalf("len(perToken) = %d, want %d", len(perToken), nTokens)
	}

	sum := 0.0
	for _, nll := range perToken {
		sum += float64(nll)
	}
	perTokenMean := float32(sum / float64(len(perToken)))
	if diff := math.Abs(float64(meanLoss - perTokenMean)); diff > tol {
		t.Fatalf("mean loss mismatch: scalar=%g per_token_mean=%g diff=%g", meanLoss, perTokenMean, diff)
	}
}
