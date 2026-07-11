//go:build mlx && cgo

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMaskedMarginPLLMatchesCPUOracleAndHasFiniteGradients(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		rows   = 4
		seqLen = 1
		vocab  = 3
	)
	prog := ir.NewProgram(1)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{rows})
	prog.DeclareInput("targets", ir.TensorInt32, []int{rows})
	prog.DeclareInput("mask", ir.TensorFloat32, []int{rows})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("rank", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("anchor", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("delta", ir.TensorFloat32, []int{1})
	prog.Embed("w0", "tokens", "logits")
	prog.MaskedMarginPLL("logits", "targets", "mask", seqLen, 1, 0.5, "loss", "rank", "anchor", "delta")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	weights := []float32{
		80, -80, -80, // preferred: active near-degenerate target 0
		0, 0, 0, // contrast: active uniform target 1
		2, 0, -1, // inactive pair must not affect the result
		0, 1, 0,
	}
	weight, err := FromDataShape(weights, []int{rows, vocab})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)
	inputs := []TensorInput{
		{Name: "tokens", DType: TensorInt32, Shape: []int{rows}, Data: []int32{0, 1, 2, 3}},
		{Name: "targets", DType: TensorInt32, Shape: []int{rows}, Data: []int32{0, 1, 0, 1}},
		{Name: "mask", DType: TensorFloat32, Shape: []int{rows}, Data: []float32{1, 1, 0, 0}},
	}
	pos := logSoftmaxTargetOracle(weights[:vocab], 0)
	neg := logSoftmaxTargetOracle(weights[vocab:2*vocab], 1)
	delta := pos - neg
	rank := stableSoftplusOracle(1 - delta)
	anchor := -pos
	wantLoss := rank + 0.5*anchor
	for name, want := range map[string]float64{
		"loss": wantLoss, "rank": rank, "anchor": anchor, "delta": delta,
	} {
		got, err := EvalProgramOutput(gpuProg, []int64{weight}, inputs, name)
		if err != nil {
			t.Fatalf("EvalProgramOutput(%s): %v", name, err)
		}
		if len(got) != 1 || math.Abs(float64(got[0])-want) > 1e-5 {
			t.Fatalf("%s=%v want %g", name, got, want)
		}
	}
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, []int64{weight}, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss=%g, want finite", loss)
	}
	for i, grad := range grads[0] {
		if math.IsNaN(float64(grad)) || math.IsInf(float64(grad), 0) {
			t.Fatalf("gradient[%d]=%g, want finite", i, grad)
		}
	}
}

func logSoftmaxTargetOracle(logits []float32, target int) float64 {
	maxValue := float64(logits[0])
	for _, value := range logits[1:] {
		maxValue = math.Max(maxValue, float64(value))
	}
	denom := 0.0
	for _, value := range logits {
		denom += math.Exp(float64(value) - maxValue)
	}
	return float64(logits[target]) - maxValue - math.Log(denom)
}

func stableSoftplusOracle(value float64) float64 {
	return math.Max(value, 0) + math.Log1p(math.Exp(-math.Abs(value)))
}
