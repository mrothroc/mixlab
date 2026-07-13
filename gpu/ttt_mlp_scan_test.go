//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestTTTMLPScanChunkOneMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, H, D, hidden, chunk = 1, 3, 1, 2, 4, 1
	q := tttPattern(B*T*H*D, 0.11)
	k := tttPattern(B*T*H*D, 0.09)
	v := tttPattern(B*T*H*D, 0.13)
	lr := []float32{-0.3, 0.1, 0.4}
	lrScale := []float32{0.75}
	tokenCoeff := []float32{0.05}
	w1 := tttPattern(H*D*hidden, 0.04)
	b1 := tttPattern(H*hidden, 0.01)
	w2 := tttPattern(H*hidden*D, 0.035)
	b2 := tttPattern(H*D, 0.008)
	normScale := []float32{1.1, 0.9}
	normBias := []float32{0.02, -0.03}

	prog := ir.NewProgram(1)
	declareTTTMLPScanTestProgram(prog, B, T, H, D, hidden, chunk)
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
	inputs := tttMLPScanInputs(B, T, H, D, hidden, chunk, q, k, v, lr, lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias)
	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuTTTMLPChunkOne(q, k, v, lr, lrScale[0], tokenCoeff[0], w1, b1, w2, b2, normScale, normBias, T, D, hidden, 0.1)
	if diff := maxAbsDiffFloat32(got, want); diff > 2e-5 {
		t.Fatalf("TTT-MLP L_inf=%g want <=2e-5\ngot=%v\nwant=%v", diff, got, want)
	}
	outs, err := EvalProgramOutputs(gpuProg, []int64{dummy}, inputs, []string{"before", "after", "update", "drift", "lr_mean", "lr_min", "lr_max"}, []int{1, 1, 1, 1, 1, 1, 1})
	if err != nil {
		t.Fatalf("EvalProgramOutputs diagnostics: %v", err)
	}
	for name, values := range outs {
		if len(values) != 1 || math.IsNaN(float64(values[0])) || math.IsInf(float64(values[0]), 0) {
			t.Fatalf("diagnostic %s=%v is not one finite scalar", name, values)
		}
	}
}

func TestTTTMLPScanChunkedRaggedTailMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, H, D, hidden, chunk = 1, 5, 1, 2, 4, 3
	q := tttPattern(B*T*H*D, 0.08)
	k := tttPattern(B*T*H*D, 0.07)
	v := tttPattern(B*T*H*D, 0.09)
	lr := []float32{-0.5, -0.1, 0.25, 0.45, 0.7}
	lrScale := []float32{0.6}
	tokenCoeff := []float32{0.03, -0.02, 0.04}
	w1 := tttPattern(H*D*hidden, 0.035)
	b1 := tttPattern(H*hidden, 0.012)
	w2 := tttPattern(H*hidden*D, 0.03)
	b2 := tttPattern(H*D, 0.007)
	normScale := []float32{1.05, 0.95}
	normBias := []float32{0.015, -0.025}

	prog := ir.NewProgram(1)
	declareTTTMLPScanTestProgram(prog, B, T, H, D, hidden, chunk)
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
	inputs := tttMLPScanInputs(B, T, H, D, hidden, chunk, q, k, v, lr, lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias)
	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuTTTMLPChunked(q, k, v, lr, lrScale[0], tokenCoeff, w1, b1, w2, b2, normScale, normBias, T, D, hidden, chunk, 0.1)
	if diff := maxAbsDiffFloat32(got, want); diff > 3e-5 {
		t.Fatalf("chunked TTT-MLP L_inf=%g want <=3e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func TestTTTMLPScanRaggedChunkAndRowResetAreFinite(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, H, D, hidden, chunk = 2, 5, 2, 2, 4, 3
	rowQ := tttPattern(T*H*D, 0.07)
	rowK := tttPattern(T*H*D, 0.06)
	rowV := tttPattern(T*H*D, 0.08)
	q := append(append([]float32(nil), rowQ...), rowQ...)
	k := append(append([]float32(nil), rowK...), rowK...)
	v := append(append([]float32(nil), rowV...), rowV...)
	lrRow := tttPattern(T*H, 0.1)
	lr := append(append([]float32(nil), lrRow...), lrRow...)
	tokenCoeff := []float32{0, 0, 0}
	w1 := tttPattern(H*D*hidden, 0.03)
	b1 := make([]float32, H*hidden)
	w2 := tttPattern(H*hidden*D, 0.025)
	b2 := make([]float32, H*D)
	normScale := []float32{1, 1, 1, 1}
	normBias := []float32{0, 0, 0, 0}

	prog := ir.NewProgram(1)
	declareTTTMLPScanTestProgram(prog, B, T, H, D, hidden, chunk)
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, _ := FromData([]float32{0}, 1, 1)
	defer FreeHandle(dummy)
	inputs := tttMLPScanInputs(B, T, H, D, hidden, chunk, q, k, v, lr, []float32{1}, tokenCoeff, w1, b1, w2, b2, normScale, normBias)
	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	rowSize := T * H * D
	for i, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("out[%d]=%g non-finite", i, value)
		}
	}
	if diff := maxAbsDiffFloat32(got[:rowSize], got[rowSize:]); diff > 1e-6 {
		t.Fatalf("identical rows diverged by %g; recurrent state leaked across rows", diff)
	}
}

func declareTTTMLPScanTestProgram(p *ir.Program, B, T, H, D, hidden, chunk int) {
	p.DeclareInput("q", ir.TensorFloat32, []int{B * T, H * D})
	p.DeclareInput("k", ir.TensorFloat32, []int{B * T, H * D})
	p.DeclareInput("v", ir.TensorFloat32, []int{B * T, H * D})
	p.DeclareInput("lr", ir.TensorFloat32, []int{B * T, H})
	p.DeclareInput("lr_scale", ir.TensorFloat32, []int{1})
	p.DeclareInput("token_coeff", ir.TensorFloat32, []int{chunk})
	p.DeclareInput("w1", ir.TensorFloat32, []int{H * D, hidden})
	p.DeclareInput("b1", ir.TensorFloat32, []int{H, hidden})
	p.DeclareInput("w2", ir.TensorFloat32, []int{H * hidden, D})
	p.DeclareInput("b2", ir.TensorFloat32, []int{H, D})
	p.DeclareInput("norm_scale", ir.TensorFloat32, []int{H, D})
	p.DeclareInput("norm_bias", ir.TensorFloat32, []int{H, D})
	p.DeclareOutput("out", ir.TensorFloat32, []int{B * T * H, D})
	for _, name := range []string{"before", "after", "update", "drift", "lr_mean", "lr_min", "lr_max"} {
		p.DeclareOutput(name, ir.TensorFloat32, []int{1})
	}
	p.TTTMLPScan("q", "k", "v", "lr", "lr_scale", "token_coeff", "w1", "b1", "w2", "b2", "norm_scale", "norm_bias", "out", "before", "after", "update", "drift", "lr_mean", "lr_min", "lr_max", B, T, H, D, hidden, chunk, 0.1)
}

func tttMLPScanInputs(B, T, H, D, hidden, chunk int, q, k, v, lr, lrScale, tokenCoeff, w1, b1, w2, b2, normScale, normBias []float32) []TensorInput {
	return []TensorInput{
		{Name: "q", DType: TensorFloat32, Shape: []int{B * T, H * D}, Data: q},
		{Name: "k", DType: TensorFloat32, Shape: []int{B * T, H * D}, Data: k},
		{Name: "v", DType: TensorFloat32, Shape: []int{B * T, H * D}, Data: v},
		{Name: "lr", DType: TensorFloat32, Shape: []int{B * T, H}, Data: lr},
		{Name: "lr_scale", DType: TensorFloat32, Shape: []int{1}, Data: lrScale},
		{Name: "token_coeff", DType: TensorFloat32, Shape: []int{chunk}, Data: tokenCoeff},
		{Name: "w1", DType: TensorFloat32, Shape: []int{H * D, hidden}, Data: w1},
		{Name: "b1", DType: TensorFloat32, Shape: []int{H, hidden}, Data: b1},
		{Name: "w2", DType: TensorFloat32, Shape: []int{H * hidden, D}, Data: w2},
		{Name: "b2", DType: TensorFloat32, Shape: []int{H, D}, Data: b2},
		{Name: "norm_scale", DType: TensorFloat32, Shape: []int{H, D}, Data: normScale},
		{Name: "norm_bias", DType: TensorFloat32, Shape: []int{H, D}, Data: normBias},
	}
}

func cpuTTTMLPChunkOne(q, k, v, lr []float32, lrScale, tokenCoeff float32, w1, b1, w2, b2, normScale, normBias []float32, T, D, hidden int, baseLR float32) []float32 {
	return cpuTTTMLPChunked(q, k, v, lr, lrScale, []float32{tokenCoeff}, w1, b1, w2, b2, normScale, normBias, T, D, hidden, 1, baseLR)
}

func cpuTTTMLPChunked(q, k, v, lr []float32, lrScale float32, tokenCoeff, w1, b1, w2, b2, normScale, normBias []float32, T, D, hidden, chunk int, baseLR float32) []float32 {
	stateW1 := append([]float32(nil), w1...)
	stateB1 := append([]float32(nil), b1...)
	stateW2 := append([]float32(nil), w2...)
	stateB2 := append([]float32(nil), b2...)
	out := make([]float32, T*D)
	for start := 0; start < T; start += chunk {
		end := min(T, start+chunk)
		K := end - start
		qChunk := append([]float32(nil), q[start*D:end*D]...)
		kChunk := append([]float32(nil), k[start*D:end*D]...)
		cpuTTTApplyChunkRoPE(qChunk, K, D)
		cpuTTTApplyChunkRoPE(kChunk, K, D)

		x2ByToken := make([][]float32, K)
		gradZ1ByToken := make([][]float32, K)
		gradZ2ByToken := make([][]float32, K)
		for local := 0; local < K; local++ {
			kt := kChunk[local*D : (local+1)*D]
			vt := v[(start+local)*D : (start+local+1)*D]
			z1 := cpuVecMat(kt, stateW1, D, hidden, stateB1)
			x2 := make([]float32, hidden)
			for i := range x2 {
				x2[i] = cpuTTTGELU(z1[i])
			}
			z2 := cpuVecMat(x2, stateW2, hidden, D, stateB2)
			target := make([]float32, D)
			for i := range target {
				target[i] = vt[i] - kt[i]
			}
			gradZ2 := cpuLayerNormL2VJP(z2, target, normScale, normBias)
			gradZ1 := make([]float32, hidden)
			for i := 0; i < hidden; i++ {
				for j := 0; j < D; j++ {
					gradZ1[i] += gradZ2[j] * stateW2[i*D+j]
				}
				gradZ1[i] *= cpuTTTGELUDerivative(z1[i])
			}
			x2ByToken[local] = x2
			gradZ1ByToken[local] = gradZ1
			gradZ2ByToken[local] = gradZ2
		}

		for local := 0; local < K; local++ {
			queryW1 := append([]float32(nil), stateW1...)
			queryB1 := append([]float32(nil), stateB1...)
			queryW2 := append([]float32(nil), stateW2...)
			queryB2 := append([]float32(nil), stateB2...)
			positionScale := max(float32(1)/float32(local+1)+tokenCoeff[local], 0)
			for source := 0; source <= local; source++ {
				gate := float32(1 / (1 + math.Exp(-float64(lr[start+source]))))
				eta := positionScale * gate * baseLR / float32(D) * lrScale
				cpuTTTApplyStateGradient(queryW1, queryB1, queryW2, queryB2,
					kChunk[source*D:(source+1)*D], x2ByToken[source],
					gradZ1ByToken[source], gradZ2ByToken[source], eta, D, hidden)
			}
			qt := qChunk[local*D : (local+1)*D]
			qz1 := cpuVecMat(qt, queryW1, D, hidden, queryB1)
			qx2 := make([]float32, hidden)
			for i := range qx2 {
				qx2[i] = cpuTTTGELU(qz1[i])
			}
			qz2 := cpuVecMat(qx2, queryW2, hidden, D, queryB2)
			ln := cpuTTTLayerNorm(qz2, normScale, normBias)
			for i := 0; i < D; i++ {
				out[(start+local)*D+i] = qt[i] + ln[i]
			}
			if local == K-1 {
				stateW1, stateB1, stateW2, stateB2 = queryW1, queryB1, queryW2, queryB2
			}
		}
	}
	return out
}

func cpuTTTApplyStateGradient(w1, b1, w2, b2, k, x2, gradZ1, gradZ2 []float32, eta float32, D, hidden int) {
	for i := 0; i < D; i++ {
		for j := 0; j < hidden; j++ {
			w1[i*hidden+j] -= eta * k[i] * gradZ1[j]
		}
	}
	for j := 0; j < hidden; j++ {
		b1[j] -= eta * gradZ1[j]
	}
	for i := 0; i < hidden; i++ {
		for j := 0; j < D; j++ {
			w2[i*D+j] -= eta * x2[i] * gradZ2[j]
		}
	}
	for j := 0; j < D; j++ {
		b2[j] -= eta * gradZ2[j]
	}
}

func cpuTTTApplyChunkRoPE(x []float32, K, D int) {
	for position := 0; position < K; position++ {
		for pair := 0; pair < D/2; pair++ {
			angle := float64(position) * math.Exp(float64(pair)*(-math.Log(10000)*2/float64(D)))
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			even := x[position*D+2*pair]
			odd := x[position*D+2*pair+1]
			x[position*D+2*pair] = even*c - odd*s
			x[position*D+2*pair+1] = even*s + odd*c
		}
	}
}

func cpuVecMat(x, w []float32, rows, cols int, bias []float32) []float32 {
	out := append([]float32(nil), bias[:cols]...)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[j] += x[i] * w[i*cols+j]
		}
	}
	return out
}

func cpuTTTLayerNorm(x, scale, bias []float32) []float32 {
	mean := float32(0)
	for _, value := range x {
		mean += value
	}
	mean /= float32(len(x))
	variance := float32(0)
	for _, value := range x {
		variance += (value - mean) * (value - mean)
	}
	variance /= float32(len(x))
	inv := float32(1 / math.Sqrt(float64(variance+1e-6)))
	out := make([]float32, len(x))
	for i := range out {
		out[i] = (x[i]-mean)*inv*scale[i] + bias[i]
	}
	return out
}

func cpuLayerNormL2VJP(x, target, scale, bias []float32) []float32 {
	mean := float32(0)
	for _, value := range x {
		mean += value
	}
	mean /= float32(len(x))
	variance := float32(0)
	for _, value := range x {
		variance += (value - mean) * (value - mean)
	}
	variance /= float32(len(x))
	std := float32(math.Sqrt(float64(variance + 1e-6)))
	xhat := make([]float32, len(x))
	gradHat := make([]float32, len(x))
	sumGrad, sumGradX := float32(0), float32(0)
	for i := range x {
		xhat[i] = (x[i] - mean) / std
		gradHat[i] = (scale[i]*xhat[i] + bias[i] - target[i]) * scale[i]
		sumGrad += gradHat[i]
		sumGradX += gradHat[i] * xhat[i]
	}
	out := make([]float32, len(x))
	for i := range out {
		out[i] = (float32(len(x))*gradHat[i] - sumGrad - xhat[i]*sumGradX) / (float32(len(x)) * std)
	}
	return out
}

func cpuTTTGELU(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Tanh(float64(0.7978845608*(x+0.044715*x*x*x)))))
}

func cpuTTTGELUDerivative(x float32) float32 {
	t := float32(math.Tanh(float64(0.79788456 * x * (1 + 0.044715*x*x))))
	return 0.5*x*((1-t*t)*(0.79788456+0.1070322243*x*x)) + 0.5*(1+t)
}

func tttPattern(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = scale * float32(math.Sin(float64(i+1))*0.7+math.Cos(float64(2*i+1))*0.3)
	}
	return out
}
