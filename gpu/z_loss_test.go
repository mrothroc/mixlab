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
