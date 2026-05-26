//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"sort"
	"strconv"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestMoEFeedForwardMatchesCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	for _, topK := range []int{1, 2} {
		t.Run("topk", func(t *testing.T) {
			const (
				B   = 1
				T   = 4
				D   = 3
				E   = 3
				FFN = 4
			)
			x := patternedFloats(B*T*D, 0.17)
			router := patternedFloats(D*E, 0.23)
			weights := [][]float32{router}
			for e := 0; e < E; e++ {
				weights = append(weights,
					patternedFloats(D*FFN, 0.09+float64(e)*0.03),
					patternedFloats(D*FFN, 0.11+float64(e)*0.03),
					patternedFloats(FFN*D, 0.13+float64(e)*0.03),
				)
			}
			handles, cleanup := uploadMoETestWeights(t, weights, []int{D, E}, []int{D, FFN}, []int{FFN, D})
			defer cleanup()

			inputs := []string{"x", "w0"}
			for i := 1; i < len(weights); i++ {
				inputs = append(inputs, weightNameForTest(i))
			}
			prog := ir.NewProgram(len(weights))
			prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
			prog.DeclareOutput("delta", ir.TensorFloat32, []int{B * T, D})
			prog.DeclareOutput("aux", ir.TensorFloat32, []int{1})
			prog.DeclareOutput("entropy", ir.TensorFloat32, []int{1})
			prog.MoEFeedForward(inputs, "delta", "aux", "entropy", B, T, D, E, topK, 0, FFN, 0, 0)

			gpuProg, err := LowerIRProgram(prog)
			if err != nil {
				t.Fatalf("LowerIRProgram: %v", err)
			}
			defer gpuProg.Destroy()
			tensorInputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x}}
			gotDelta, err := EvalProgramOutput(gpuProg, handles, tensorInputs, "delta")
			if err != nil {
				t.Fatalf("EvalProgramOutput(delta): %v", err)
			}
			gotAux, err := EvalProgramOutput(gpuProg, handles, tensorInputs, "aux")
			if err != nil {
				t.Fatalf("EvalProgramOutput(aux): %v", err)
			}
			gotEntropy, err := EvalProgramOutput(gpuProg, handles, tensorInputs, "entropy")
			if err != nil {
				t.Fatalf("EvalProgramOutput(entropy): %v", err)
			}
			wantDelta, wantAux, wantEntropy := cpuMoESwiGLU(x, weights, B*T, D, E, topK, FFN)
			if diff := maxAbsDiffFloat32(gotDelta, wantDelta); diff > 2e-5 {
				t.Fatalf("delta L_inf=%g, want <= 2e-5\ngot=%v\nwant=%v", diff, gotDelta, wantDelta)
			}
			if diff := math.Abs(float64(gotAux[0] - wantAux)); diff > 2e-5 {
				t.Fatalf("aux=%g, want %g diff=%g", gotAux[0], wantAux, diff)
			}
			if diff := math.Abs(float64(gotEntropy[0] - wantEntropy)); diff > 2e-5 {
				t.Fatalf("entropy=%g, want %g diff=%g", gotEntropy[0], wantEntropy, diff)
			}
		})
	}
}

func TestMoEFeedForwardGradientsFinite(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B   = 1
		T   = 3
		D   = 3
		E   = 2
		FFN = 4
		V   = 5
	)
	x := patternedFloats(B*T*D, 0.17)
	targets := []int32{0, 2, 4}
	router := patternedFloats(D*E, 0.23)
	weights := [][]float32{router}
	for e := 0; e < E; e++ {
		weights = append(weights,
			patternedFloats(D*FFN, 0.09+float64(e)*0.03),
			patternedFloats(D*FFN, 0.11+float64(e)*0.03),
			patternedFloats(FFN*D, 0.13+float64(e)*0.03),
		)
	}
	weights = append(weights, patternedFloats(D*V, 0.19))
	handles, cleanup := uploadMoETestWeights(t, weights, []int{D, E}, []int{D, FFN}, []int{FFN, D}, []int{D, V})
	defer cleanup()

	inputs := []string{"x", "w0"}
	for i := 1; i < len(weights)-1; i++ {
		inputs = append(inputs, weightNameForTest(i))
	}
	prog := ir.NewProgram(len(weights))
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareInput("targets", ir.TensorInt32, []int{B * T})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.MoEFeedForward(inputs, "delta", "aux", "entropy", B, T, D, E, 1, 0, FFN, 0, 0)
	prog.MatMul("delta", weightNameForTest(len(weights)-1), "logits")
	prog.CrossEntropy("logits", "targets", "ce")
	prog.ScalarMul("aux", 0.01, "aux_weighted")
	prog.Add("ce", "aux_weighted", "loss")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	loss, grads, err := EvalProgramGradientsForOutput(gpuProg, handles, []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x},
		{Name: "targets", DType: TensorInt32, Shape: []int{B * T}, Data: targets},
	}, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss=%g, want finite", loss)
	}
	for i, grad := range grads {
		for j, v := range grad {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("grad[%d][%d]=%g, want finite", i, j, v)
			}
		}
	}
	if maxAbsFloat32(grads[0]) == 0 {
		t.Fatal("router gradient is zero, want non-zero signal")
	}
}

func uploadMoETestWeights(t *testing.T, weights [][]float32, shapes ...[]int) ([]int64, func()) {
	t.Helper()
	handles := make([]int64, 0, len(weights))
	for i, w := range weights {
		shape := shapes[len(shapes)-1]
		switch {
		case i == 0:
			shape = shapes[0]
		case i == len(weights)-1 && len(shapes) > 3:
			shape = shapes[len(shapes)-1]
		case (i-1)%3 == 2:
			shape = shapes[2]
		default:
			shape = shapes[1]
		}
		h, err := FromDataShape(w, shape)
		if err != nil {
			for _, old := range handles {
				FreeHandle(old)
			}
			t.Fatalf("FromData(weight %d): %v", i, err)
		}
		handles = append(handles, h)
	}
	return handles, func() {
		for _, h := range handles {
			FreeHandle(h)
		}
	}
}

func weightNameForTest(i int) string {
	return "w" + strconv.Itoa(i)
}

func cpuMoESwiGLU(x []float32, weights [][]float32, rows, D, E, topK, ffn int) ([]float32, float32, float32) {
	router := weights[0]
	logits := matmulCPU(x, router, rows, D, E)
	probs := softmaxRowsCPU(logits, rows, E)
	out := make([]float32, rows*D)
	selectedCounts := make([]float32, E)
	probMeans := make([]float32, E)
	var entropy float64
	for r := 0; r < rows; r++ {
		order := make([]int, E)
		for e := 0; e < E; e++ {
			order[e] = e
			probMeans[e] += probs[r*E+e]
			p := float64(probs[r*E+e])
			if p > 0 {
				entropy -= p * math.Log(p)
			}
		}
		sort.Slice(order, func(i, j int) bool {
			return probs[r*E+order[i]] > probs[r*E+order[j]]
		})
		var denom float32
		for k := 0; k < topK; k++ {
			denom += probs[r*E+order[k]]
		}
		for k := 0; k < topK; k++ {
			e := order[k]
			selectedCounts[e] += 1 / float32(topK)
			gate := probs[r*E+e] / denom
			base := 1 + e*3
			gateProj := matmulRowCPU(x[r*D:(r+1)*D], weights[base], D, ffn)
			upProj := matmulRowCPU(x[r*D:(r+1)*D], weights[base+1], D, ffn)
			ff := make([]float32, ffn)
			for j := 0; j < ffn; j++ {
				ff[j] = sigmoid32(gateProj[j]) * upProj[j]
			}
			expertOut := matmulRowCPU(ff, weights[base+2], ffn, D)
			for d := 0; d < D; d++ {
				out[r*D+d] += gate * expertOut[d]
			}
		}
	}
	var aux float32
	for e := 0; e < E; e++ {
		probMean := probMeans[e] / float32(rows)
		selectedFraction := selectedCounts[e] / float32(rows)
		aux += probMean * selectedFraction
	}
	return out, aux * float32(E), float32(entropy / float64(rows))
}

func matmulCPU(a, b []float32, rows, inner, cols int) []float32 {
	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			var sum float32
			for k := 0; k < inner; k++ {
				sum += a[r*inner+k] * b[k*cols+c]
			}
			out[r*cols+c] = sum
		}
	}
	return out
}

func matmulRowCPU(a, b []float32, inner, cols int) []float32 {
	out := make([]float32, cols)
	for c := 0; c < cols; c++ {
		var sum float32
		for k := 0; k < inner; k++ {
			sum += a[k] * b[k*cols+c]
		}
		out[c] = sum
	}
	return out
}

func softmaxRowsCPU(logits []float32, rows, cols int) []float32 {
	out := make([]float32, len(logits))
	for r := 0; r < rows; r++ {
		maxV := float32(math.Inf(-1))
		for c := 0; c < cols; c++ {
			if logits[r*cols+c] > maxV {
				maxV = logits[r*cols+c]
			}
		}
		var denom float64
		for c := 0; c < cols; c++ {
			v := math.Exp(float64(logits[r*cols+c] - maxV))
			out[r*cols+c] = float32(v)
			denom += v
		}
		for c := 0; c < cols; c++ {
			out[r*cols+c] /= float32(denom)
		}
	}
	return out
}

func sigmoid32(v float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-v))))
}

func maxAbsFloat32(values []float32) float32 {
	var maxV float32
	for _, v := range values {
		if v < 0 {
			v = -v
		}
		if v > maxV {
			maxV = v
		}
	}
	return maxV
}
