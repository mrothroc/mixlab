//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestFirstByteMaskedCrossEntropyNoHigherThanUnmasked(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		nTokens = 3
		vocab   = 5
		tol     = 1e-6
	)

	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{nTokens, vocab})
	prog.DeclareInput("targets", ir.TensorInt32, []int{nTokens})
	prog.DeclareInput("first_byte_valid", ir.TensorInt32, []int{vocab})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("eval_loss", ir.TensorFloat32, []int{1})
	prog.FirstByteMaskedCrossEntropy("logits", "targets", "first_byte_valid", "loss")
	prog.CrossEntropy("logits", "targets", "eval_loss")

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
				1.0, 0.2, -0.5, 2.0, 1.5,
				0.1, 0.4, 1.2, -0.6, 2.5,
				2.0, -0.2, 0.1, 3.0, 1.0,
			},
		},
		{Name: "targets", DType: TensorInt32, Shape: []int{nTokens}, Data: []int32{0, 3, 2}},
		{Name: "first_byte_valid", DType: TensorInt32, Shape: []int{vocab}, Data: []int32{1, 1, 1, 0, 0}},
	}

	maskedLoss, err := TrainerEvaluate(trainer, inputs)
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}
	unmasked, err := TrainerReadOutput(trainer, "eval_loss", []int{1})
	if err != nil {
		t.Fatalf("TrainerReadOutput(eval_loss): %v", err)
	}
	if len(unmasked) != 1 {
		t.Fatalf("eval_loss len=%d, want 1", len(unmasked))
	}
	if math.IsNaN(float64(maskedLoss)) || math.IsInf(float64(maskedLoss), 0) {
		t.Fatalf("masked loss is not finite: %g", maskedLoss)
	}
	if maskedLoss > unmasked[0]+tol {
		t.Fatalf("masked loss=%g exceeds unmasked=%g", maskedLoss, unmasked[0])
	}
}
