//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedCrossEntropyIgnoresFiniteExtremeInactiveRows(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows  = 2
		vocab = 3
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("per_token", ir.TensorFloat32, []int{rows})
	prog.MaskedCrossEntropy("w0", "targets", "mask", "loss")
	prog.MaskedCrossEntropyPerToken("w0", "targets", "mask", "per_token")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	weights := []float32{
		0, 1, -1,
		math.MaxFloat32, -math.MaxFloat32, 0,
	}
	weight, err := FromDataShape(weights, []int{rows, vocab})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)
	inputs := []TensorInput{
		{Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: []int32{1, 1}},
		{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: []float32{1, 0}},
	}
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, []int64{weight}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	want := float32(-logSoftmaxTargetOracle(weights[:vocab], 1))
	if !isFiniteOptimizerGuardValue(loss) || math.Abs(float64(loss-want)) > 1e-6 {
		t.Fatalf("loss=%g want %g", loss, want)
	}
	for i, grad := range grads[0] {
		if !isFiniteOptimizerGuardValue(grad) {
			t.Fatalf("gradient[%d]=%g, want finite", i, grad)
		}
		if i >= vocab && grad != 0 {
			t.Fatalf("inactive-row gradient[%d]=%g, want 0", i, grad)
		}
	}
	perToken, err := EvalProgramOutput(gpuProg, []int64{weight}, inputs, "per_token")
	if err != nil {
		t.Fatalf("EvalProgramOutput(per_token): %v", err)
	}
	if len(perToken) != rows || perToken[1] != 0 || !isFiniteOptimizerGuardValue(perToken[0]) {
		t.Fatalf("per_token=%v, want [finite, 0]", perToken)
	}
}

func TestPLLMarginPairRowsDoNotPoisonPrimaryOrZLoss(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows   = 4
		seqLen = 1
		vocab  = 3
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareInput("primary_mask", ir.TensorFloat32, []int{rows})
	prog.DeclareInput("pll_mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.MaskedCrossEntropy("w0", "targets", "primary_mask", "primary_loss")
	prog.MaskedMarginPLL("w0", "targets", "pll_mask", seqLen, 1, 0, "pll_loss", "rank", "anchor", "delta")
	prog.MaskedZLoss("w0", "primary_mask", "z_loss")
	prog.ScalarMul("pll_loss", 0.1, "weighted_pll")
	prog.ScalarMul("z_loss", 0.0001, "weighted_z")
	prog.Add("primary_loss", "weighted_pll", "task_plus_pll")
	prog.Add("task_plus_pll", "weighted_z", "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	weights := []float32{
		math.MaxFloat32, -math.MaxFloat32, 0,
		-math.MaxFloat32, math.MaxFloat32, 0,
		0, 1, -1,
		1, 0, -1,
	}
	weight, err := FromDataShape(weights, []int{rows, vocab})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)
	inputs := []TensorInput{
		{Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: []int32{1, 1, 1, 0}},
		{Name: "primary_mask", DType: TensorFloat32, Shape: []int{rows}, Data: []float32{0, 0, 1, 1}},
		{Name: "pll_mask", DType: TensorFloat32, Shape: []int{rows}, Data: []float32{1, 1, 0, 0}},
	}
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, []int64{weight}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	if !isFiniteOptimizerGuardValue(loss) {
		t.Fatalf("combined loss=%g, want finite", loss)
	}
	for i, grad := range grads[0] {
		if !isFiniteOptimizerGuardValue(grad) {
			t.Fatalf("combined gradient[%d]=%g, want finite", i, grad)
		}
	}
}
