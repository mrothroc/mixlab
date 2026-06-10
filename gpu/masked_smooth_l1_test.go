//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedSmoothL1MatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows = 3
		dim  = 2
		beta = 1.0
		tol  = 1e-5
	)
	pred := []float32{0, 2, 3, -1, -2, 0.25}
	target := []float32{1, 0, 3, -3, 0, -0.25}
	mask := []float32{1, 0, 1}

	prog := ir.NewProgram(1)
	prog.DeclareInput("pred", ir.TensorFloat32, []int{rows, dim})
	prog.DeclareInput("target", ir.TensorFloat32, []int{rows, dim})
	prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.MaskedSmoothL1("pred", "target", "mask", beta, "loss")

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
	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "pred", DType: TensorFloat32, Shape: []int{rows, dim}, Data: pred},
		{Name: "target", DType: TensorFloat32, Shape: []int{rows, dim}, Data: target},
		{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: mask},
	}, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := maskedSmoothL1Oracle(pred, target, mask, dim, beta)
	if diff := math.Abs(float64(got[0] - want)); diff > tol {
		t.Fatalf("loss=%g want=%g diff=%g", got[0], want, diff)
	}

	zeroMask := []float32{0, 0, 0}
	got, err = EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "pred", DType: TensorFloat32, Shape: []int{rows, dim}, Data: pred},
		{Name: "target", DType: TensorFloat32, Shape: []int{rows, dim}, Data: target},
		{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: zeroMask},
	}, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput zero mask: %v", err)
	}
	if got[0] != 0 {
		t.Fatalf("zero mask loss=%g want 0", got[0])
	}
}

func TestEvalProgramOutputsReadsMultipleNamedOutputs(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("a", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("b", ir.TensorFloat32, []int{2})
	prog.ScalarMul("x", 2, "a")
	prog.ScalarMul("x", -1, "b")

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

	outs, err := EvalProgramOutputs(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{2}, Data: []float32{3, -4}},
	}, []string{"a", "b"}, []int{2, 2})
	if err != nil {
		t.Fatalf("EvalProgramOutputs: %v", err)
	}
	if got := outs["a"]; len(got) != 2 || got[0] != 6 || got[1] != -8 {
		t.Fatalf("a=%v, want [6 -8]", got)
	}
	if got := outs["b"]; len(got) != 2 || got[0] != -3 || got[1] != 4 {
		t.Fatalf("b=%v, want [-3 4]", got)
	}
}

func maskedSmoothL1Oracle(pred, target, mask []float32, dim int, beta float64) float32 {
	var sum float64
	var active float64
	for row := 0; row < len(mask); row++ {
		if mask[row] <= 0 {
			continue
		}
		active++
		for j := 0; j < dim; j++ {
			idx := row*dim + j
			diff := math.Abs(float64(pred[idx] - target[idx]))
			if diff < beta {
				sum += 0.5 * diff * diff / beta
			} else {
				sum += diff - 0.5*beta
			}
		}
	}
	if active == 0 {
		return 0
	}
	return float32(sum / (active * float64(dim)))
}
