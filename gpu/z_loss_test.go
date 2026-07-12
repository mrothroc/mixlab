//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestZLossMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows  = 3
		vocab = 4
		tol   = 1e-5
	)
	logits := []float32{
		1.0, -2.0, 0.5, 3.0,
		0.0, 0.0, 0.0, 0.0,
		-1.5, 2.25, 0.25, -0.75,
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("logits", ir.TensorFloat32, []int{rows, vocab})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.ZLoss("logits", "loss")

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
		{Name: "logits", DType: TensorFloat32, Shape: []int{rows, vocab}, Data: logits},
	}, "loss")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := zLossOracle(logits, rows, vocab)
	if diff := math.Abs(float64(got[0] - want)); diff > tol {
		t.Fatalf("z_loss=%g want=%g diff=%g", got[0], want, diff)
	}
}

func TestMaskedZLossIgnoresFiniteExtremeInactiveRows(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows  = 2
		vocab = 3
	)
	weights := []float32{
		0, 1, -1,
		math.MaxFloat32, -math.MaxFloat32, 0,
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.MaskedZLoss("w0", "mask", "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	weight, err := FromDataShape(weights, []int{rows, vocab})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)
	inputs := []TensorInput{{
		Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: []float32{1, 0},
	}}
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, []int64{weight}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	want := zLossOracle(weights[:vocab], 1, vocab)
	if !isFiniteOptimizerGuardValue(loss) || math.Abs(float64(loss-want)) > 1e-5 {
		t.Fatalf("masked z_loss=%g want %g", loss, want)
	}
	for i, grad := range grads[0] {
		if !isFiniteOptimizerGuardValue(grad) {
			t.Fatalf("gradient[%d]=%g, want finite", i, grad)
		}
		if i >= vocab && grad != 0 {
			t.Fatalf("inactive-row gradient[%d]=%g, want 0", i, grad)
		}
	}
}

func zLossOracle(logits []float32, rows, vocab int) float32 {
	total := 0.0
	for r := 0; r < rows; r++ {
		row := logits[r*vocab : (r+1)*vocab]
		maxLogit := float64(row[0])
		for _, v := range row[1:] {
			if float64(v) > maxLogit {
				maxLogit = float64(v)
			}
		}
		sum := 0.0
		for _, v := range row {
			sum += math.Exp(float64(v) - maxLogit)
		}
		logZ := maxLogit + math.Log(sum)
		total += logZ * logZ
	}
	return float32(total / float64(rows))
}
