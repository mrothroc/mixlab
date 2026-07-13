//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTTTMLPStatefulScanMatchesChunkAndSplitDecode(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, H, D, hidden, chunk = 1, 4, 1, 2, 4, 4
	qk := tttPattern(B*T*H*D, 0.07)
	v := tttPattern(B*T*H*D, 0.09)
	lr := []float32{-0.4, -0.1, 0.2, 0.5}
	lrScale := []float32{0.8}
	tokenCoeff := []float32{0.03, -0.01, 0.02, 0.04}
	w1 := tttPattern(H*D*hidden, 0.035)
	b1 := tttPattern(H*hidden, 0.01)
	w2 := tttPattern(H*hidden*D, 0.03)
	b2 := tttPattern(H*D, 0.006)
	normScale := []float32{1.05, 0.95}
	normBias := []float32{0.01, -0.02}
	stateData := append(append(append(append([]float32(nil), w1...), b1...), w2...), b2...)
	stateSize := len(stateData)

	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandle(dummy)
	initialState, _ := FromDataShape(stateData, []int{B, stateSize})
	initialGrad, _ := FromDataShape(make([]float32, stateSize), []int{B, stateSize})
	initialConv, _ := FromDataShape(make([]float32, B*2*3*H*D), []int{B, 2, 3, H * D})
	defer FreeHandles([]int64{initialState, initialGrad, initialConv})

	full, fullState := evalTTTMLPStatefulForTest(t, dummy, 0, T, H, D, hidden, chunk,
		qk, v, lr, lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias,
		initialState, initialGrad, initialConv)
	defer FreeHandles([]int64{fullState.mlp, fullState.grad, fullState.conv})
	want := cpuTTTMLPChunked(qk, qk, v, lr, lrScale[0], tokenCoeff, w1, b1, w2, b2, normScale, normBias, T, D, hidden, chunk, 0.1)
	if diff := maxAbsDiffFloat32(full, want); diff > 4e-5 {
		t.Fatalf("stateful full chunk L_inf=%g want <=4e-5\ngot=%v\nwant=%v", diff, full, want)
	}

	first, firstState := evalTTTMLPStatefulForTest(t, dummy, 0, 2, H, D, hidden, chunk,
		qk[:2*D], v[:2*D], lr[:2], lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias,
		initialState, initialGrad, initialConv)
	second, secondState := evalTTTMLPStatefulForTest(t, dummy, 2, 2, H, D, hidden, chunk,
		qk[2*D:], v[2*D:], lr[2:], lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias,
		firstState.mlp, firstState.grad, firstState.conv)
	FreeHandles([]int64{firstState.mlp, firstState.grad, firstState.conv})
	defer FreeHandles([]int64{secondState.mlp, secondState.grad, secondState.conv})
	split := append(first, second...)
	if diff := maxAbsDiffFloat32(split, full); diff > 4e-5 {
		t.Fatalf("split cached decode L_inf=%g want <=4e-5\nsplit=%v\nfull=%v", diff, split, full)
	}
	fullPacked, err := ReadHandle(fullState.mlp)
	if err != nil {
		t.Fatal(err)
	}
	splitPacked, err := ReadHandle(secondState.mlp)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(splitPacked, fullPacked); diff > 4e-5 {
		t.Fatalf("split final state L_inf=%g want <=4e-5", diff)
	}
}

func evalTTTMLPStatefulForTest(t *testing.T, dummy int64, offset, T, H, D, hidden, chunk int,
	qk, v, lr, lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias []float32,
	state, grad, conv int64,
) ([]float32, tttMLPStateHandlesForTest) {
	t.Helper()
	const B = 1
	prog := ir.NewProgram(1)
	for name, shape := range map[string][]int{
		"qk": {B * T, H * D}, "v": {B * T, H * D}, "lr": {B * T, H},
		"q_conv_w": {H * D, 4}, "q_conv_b": {H * D}, "k_conv_w": {H * D, 4}, "k_conv_b": {H * D},
		"lr_scale": {1}, "token_coeff": {chunk}, "norm_scale": {H, D}, "norm_bias": {H, D},
		"state": {B, len(w1) + len(b1) + len(w2) + len(b2)}, "grad": {B, len(w1) + len(b1) + len(w2) + len(b2)},
		"conv": {B, 2, 3, H * D},
	} {
		prog.DeclareInput(name, ir.TensorFloat32, shape)
	}
	stateSize := len(w1) + len(b1) + len(w2) + len(b2)
	prog.DeclareOutput("out", ir.TensorFloat32, []int{B * T * H, D})
	prog.DeclareOutput("state_next", ir.TensorFloat32, []int{B, stateSize})
	prog.DeclareOutput("grad_next", ir.TensorFloat32, []int{B, stateSize})
	prog.DeclareOutput("conv_next", ir.TensorFloat32, []int{B, 2, 3, H * D})
	prog.TTTMLPStatefulScan("qk", "v", "lr", "q_conv_w", "q_conv_b", "k_conv_w", "k_conv_b",
		"lr_scale", "token_coeff", "norm_scale", "norm_bias", "state", "grad", "conv",
		"out", "state_next", "grad_next", "conv_next", B, T, H, D, hidden, chunk, offset, 0.1)
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuProg.Destroy()
	convW := make([]float32, H*D*4)
	for i := 0; i < H*D; i++ {
		convW[i*4+3] = 1
	}
	inputs := []TensorInput{
		{Name: "qk", DType: TensorFloat32, Shape: []int{B * T, H * D}, Data: qk},
		{Name: "v", DType: TensorFloat32, Shape: []int{B * T, H * D}, Data: v},
		{Name: "lr", DType: TensorFloat32, Shape: []int{B * T, H}, Data: lr},
		{Name: "q_conv_w", DType: TensorFloat32, Shape: []int{H * D, 4}, Data: convW},
		{Name: "q_conv_b", DType: TensorFloat32, Shape: []int{H * D}, Data: make([]float32, H*D)},
		{Name: "k_conv_w", DType: TensorFloat32, Shape: []int{H * D, 4}, Data: convW},
		{Name: "k_conv_b", DType: TensorFloat32, Shape: []int{H * D}, Data: make([]float32, H*D)},
		{Name: "lr_scale", DType: TensorFloat32, Shape: []int{1}, Data: lrScale},
		{Name: "token_coeff", DType: TensorFloat32, Shape: []int{chunk}, Data: tokenCoeff},
		{Name: "norm_scale", DType: TensorFloat32, Shape: []int{H, D}, Data: normScale},
		{Name: "norm_bias", DType: TensorFloat32, Shape: []int{H, D}, Data: normBias},
	}
	outs, err := EvalProgramHandleOutputs(gpuProg, []int64{dummy}, inputs, []HandleInput{
		{Name: "state", Handle: state}, {Name: "grad", Handle: grad}, {Name: "conv", Handle: conv},
	}, []string{"out", "state_next", "grad_next", "conv_next"})
	if err != nil {
		t.Fatal(err)
	}
	out, err := ReadHandle(outs["out"])
	if err != nil {
		FreeHandles([]int64{outs["out"], outs["state_next"], outs["grad_next"], outs["conv_next"]})
		t.Fatal(err)
	}
	FreeHandle(outs["out"])
	return out, tttMLPStateHandlesForTest{mlp: outs["state_next"], grad: outs["grad_next"], conv: outs["conv_next"]}
}

type tttMLPStateHandlesForTest struct {
	mlp  int64
	grad int64
	conv int64
}
