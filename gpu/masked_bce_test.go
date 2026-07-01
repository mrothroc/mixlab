//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedBCEWithLogitsAndAccuracyMatchCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows = 5
		tol  = 1e-5
	)
	logits := []float32{-2, -0.5, 0, 1.25, 3}
	targets := []int32{0, 1, 1, 1, 0}
	mask := []float32{1, 0, 1, 1, 1}

	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{rows, 1})
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("accuracy", ir.TensorFloat32, []int{1})
	prog.MaskedBCEWithLogits("logits", "targets", "mask", "loss")
	prog.MaskedBinaryAccuracy("logits", "targets", "mask", "accuracy")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)
	inputs := []TensorInput{
		{Name: "logits", DType: TensorFloat32, Shape: []int{rows, 1}, Data: logits},
		{Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: targets},
		{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: mask},
	}
	gotLoss, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput(loss): %v", err)
	}
	gotAcc, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "accuracy")
	if err != nil {
		t.Fatalf("EvalProgramOutput(accuracy): %v", err)
	}
	wantLoss, wantAcc := maskedBCEOracle(logits, targets, mask)
	if math.Abs(float64(gotLoss[0]-wantLoss)) > tol {
		t.Fatalf("loss=%g want %g", gotLoss[0], wantLoss)
	}
	if math.Abs(float64(gotAcc[0]-wantAcc)) > tol {
		t.Fatalf("accuracy=%g want %g", gotAcc[0], wantAcc)
	}
}

func maskedBCEOracle(logits []float32, targets []int32, mask []float32) (float32, float32) {
	sum := 0.0
	correct := 0.0
	count := 0.0
	for i, l := range logits {
		if mask[i] <= 0 {
			continue
		}
		y := 0.0
		if targets[i] > 0 {
			y = 1
		}
		x := float64(l)
		loss := math.Max(x, 0) - y*x + math.Log1p(math.Exp(-math.Abs(x)))
		sum += loss
		pred := int32(0)
		if l > 0 {
			pred = 1
		}
		if pred == targets[i] {
			correct++
		}
		count++
	}
	if count == 0 {
		return 0, 0
	}
	return float32(sum / count), float32(correct / count)
}
