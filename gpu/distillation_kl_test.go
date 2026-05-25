//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestDistillationKLMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		nRows = 3
		vocab = 4
		tol   = 1e-5
	)
	logits := []float32{
		1.4, -0.2, 0.1, 2.0,
		-0.5, 0.3, 1.7, -1.0,
		0.0, 0.0, 0.0, 0.0,
	}
	teacher := []float32{
		0.70, 0.00, 0.20, 0.10,
		0.05, 0.10, 0.80, 0.05,
		0.25, 0.25, 0.25, 0.25,
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{nRows, vocab})
	prog.DeclareInput("teacher_probs", ir.TensorFloat32, []int{nRows, vocab})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DistillationKL("logits", "teacher_probs", "loss")

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

	got, err := TrainerEvaluate(trainer, []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{nRows, vocab}, Data: logits},
		{Name: "teacher_probs", DType: TensorFloat32, Shape: []int{nRows, vocab}, Data: teacher},
	})
	if err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}
	want := distillationKLOracle(logits, teacher, vocab)
	if diff := math.Abs(float64(got - want)); diff > tol {
		t.Fatalf("loss = %g, want %g, diff %g", got, want, diff)
	}
}

func distillationKLOracle(logits, teacher []float32, vocab int) float32 {
	rows := len(logits) / vocab
	var total float64
	for r := 0; r < rows; r++ {
		start := r * vocab
		end := start + vocab
		maxVal := logits[start]
		for _, v := range logits[start+1 : end] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float64
		for _, v := range logits[start:end] {
			sum += math.Exp(float64(v - maxVal))
		}
		logNorm := float64(maxVal) + math.Log(sum)
		var rowKL float64
		for i := start; i < end; i++ {
			p := float64(teacher[i])
			if p == 0 {
				continue
			}
			studentLogProb := float64(logits[i]) - logNorm
			rowKL += p * (math.Log(p) - studentLogProb)
		}
		total += rowKL
	}
	return float32(total / float64(rows))
}
