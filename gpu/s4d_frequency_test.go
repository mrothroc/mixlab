//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"math/cmplx"
	"runtime"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestS4DSobolevFilterMatchesCPUOracleAndHasFiniteGradients(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N = 1, 5, 2, 4
	x, weights := s4dParityFixture(B, T, D, N)
	weights = append(weights, []float32{-0.5, 1.25})

	prog := lowerS4DSobolevProgram(t, B, T, D, N)
	defer prog.Destroy()
	handles := s4dSobolevWeightHandles(t, weights, D, N)
	defer FreeHandles(handles)
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x}}

	got, err := EvalProgramOutput(prog, handles, inputs, "output")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	want := cpuS4DSobolevForward(x, weights, B, T, D, N)
	if diff := maxAbsDiffFloat32(got, want); diff > 2e-5 {
		t.Fatalf("Sobolev S4D CPU parity L_inf=%g want <=2e-5\ngot=%v\nwant=%v", diff, got, want)
	}

	_, grads, err := EvalProgramGradientsForOutput(prog, handles, inputs, "loss")
	if err != nil {
		t.Fatalf("EvalProgramGradientsForOutput: %v", err)
	}
	if len(grads) != len(weights) {
		t.Fatalf("gradient tensors=%d want %d", len(grads), len(weights))
	}
	betaGradientNonzero := false
	for weightIndex, grad := range grads {
		for valueIndex, value := range grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("gradient[%d][%d]=%g is non-finite", weightIndex, valueIndex, value)
			}
			if weightIndex == len(grads)-1 && math.Abs(float64(value)) > 1e-8 {
				betaGradientNonzero = true
			}
		}
	}
	if !betaGradientNonzero {
		t.Fatal("Sobolev beta received only zero gradients")
	}
	cpuGrads := finiteDifferenceS4DSobolevGrads(x, weights, B, T, D, N)
	for weightIndex := range grads {
		for valueIndex := range grads[weightIndex] {
			gotGrad := grads[weightIndex][valueIndex]
			wantGrad := cpuGrads[weightIndex][valueIndex]
			tolerance := float32(5e-3 + 0.08*math.Abs(float64(wantGrad)))
			if diff := float32(math.Abs(float64(gotGrad - wantGrad))); diff > tolerance {
				t.Fatalf(
					"Sobolev MLX/CPU grad w%d[%d]=%g want=%g diff=%g tolerance=%g",
					weightIndex, valueIndex, gotGrad, wantGrad, diff, tolerance,
				)
			}
		}
	}
}

func TestS4DSobolevZeroBetaMatchesUnfilteredFFT(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N = 1, 5, 2, 4
	x, weights := s4dParityFixture(B, T, D, N)
	baseline, _ := runS4DWeights(t, 0, x, weights, B, T, D, N)
	weights = append(weights, make([]float32, D))
	prog := lowerS4DSobolevProgram(t, B, T, D, N)
	defer prog.Destroy()
	handles := s4dSobolevWeightHandles(t, weights, D, N)
	defer FreeHandles(handles)
	got, err := EvalProgramOutput(prog, handles, []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}, "output")
	if err != nil {
		t.Fatal(err)
	}
	want := cpuS4DSobolevForward(x, weights, B, T, D, N)
	if diff := maxAbsDiffFloat32(got, want); diff > 2e-5 {
		t.Fatalf("zero-beta CPU-oracle output diff=%g want <=2e-5", diff)
	}
	if runtime.GOOS == "darwin" {
		if diff := maxAbsDiffFloat32(got, baseline); diff > 1e-6 {
			t.Fatalf("zero-beta Metal regression diff=%g want <=1e-6", diff)
		}
	}
}

func TestS4DAdvancedSobolevZeroBetaMatchesUnfilteredBidirectional(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N, nSSM = 1, 5, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	baseline, _ := runS4DReferenceWeights(t, x, weights, B, T, D, N, nSSM, true)

	progIR := ir.NewProgram(11)
	progIR.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	progIR.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	inputs := []string{"x", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10"}
	progIR.S4DAdvancedSobolev(
		inputs, "output", "kernel", B, T, D, N, nSSM, true,
		ir.S4DDiscretizationBilinear, true,
	)
	prog, err := LowerIRProgram(progIR)
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Destroy()
	handles := s4dReferenceWeightHandles(t, weights, D, N, nSSM)
	betaHandle, err := FromDataShape(make([]float32, D), []int{D})
	if err != nil {
		FreeHandles(handles)
		t.Fatal(err)
	}
	handles = append(handles, betaHandle)
	defer FreeHandles(handles)
	got, err := EvalProgramOutput(prog, handles, []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}, "output")
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(got, baseline); diff > 1e-6 {
		t.Fatalf("advanced zero-beta output diff=%g want <=1e-6", diff)
	}
}

func TestS4DAdvancedSobolevBidirectionalHasFiniteGradients(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N, nSSM = 1, 5, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)

	progIR := ir.NewProgram(11)
	progIR.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	progIR.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	progIR.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	inputs := []string{"x", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10"}
	progIR.S4DAdvancedSobolev(
		inputs, "output", "kernel", B, T, D, N, nSSM, true,
		ir.S4DDiscretizationBilinear, true,
	)
	progIR.Square("output", "squared")
	progIR.MeanAxis("squared", 1, "loss_rows")
	progIR.MeanAxis("loss_rows", 0, "loss")
	prog, err := LowerIRProgram(progIR)
	if err != nil {
		t.Fatal(err)
	}
	defer prog.Destroy()
	handles := s4dReferenceWeightHandles(t, weights, D, N, nSSM)
	betaHandle, err := FromDataShape(
		[]float32{-0.75, -0.25, 0.5, 1.0}, []int{D},
	)
	if err != nil {
		FreeHandles(handles)
		t.Fatal(err)
	}
	handles = append(handles, betaHandle)
	defer FreeHandles(handles)
	_, grads, err := EvalProgramGradientsForOutput(prog, handles, []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}, "loss")
	if err != nil {
		t.Fatal(err)
	}
	if len(grads) != len(handles) {
		t.Fatalf("gradient tensors=%d want %d", len(grads), len(handles))
	}
	betaNonzero := false
	for weightIndex, grad := range grads {
		for valueIndex, value := range grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("gradient[%d][%d]=%g is non-finite", weightIndex, valueIndex, value)
			}
			if weightIndex == len(grads)-1 && math.Abs(float64(value)) > 1e-8 {
				betaNonzero = true
			}
		}
	}
	if !betaNonzero {
		t.Fatal("advanced bidirectional Sobolev beta received only zero gradients")
	}
}

func lowerS4DSobolevProgram(t *testing.T, B, T, D, N int) *Program {
	t.Helper()
	prog := ir.NewProgram(7)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.S4DSobolev(
		"x", "w0", "w1", "w2", "w3", "w4", "w5", "w6",
		"output", "kernel", B, T, D, N,
	)
	prog.Square("output", "squared")
	prog.MeanAxis("squared", 1, "loss_rows")
	prog.MeanAxis("loss_rows", 0, "loss")
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	return gpuProg
}

func s4dSobolevWeightHandles(t *testing.T, weights [][]float32, D, N int) []int64 {
	t.Helper()
	shapes := [][]int{{D}, {D, N / 2}, {D, N / 2}, {D, N / 2}, {D, N / 2}, {D}, {D}}
	handles := make([]int64, len(weights))
	for i, values := range weights {
		handle, err := FromDataShape(append([]float32(nil), values...), shapes[i])
		if err != nil {
			FreeHandles(handles[:i])
			t.Fatal(err)
		}
		handles[i] = handle
	}
	return handles
}

func cpuS4DSobolevForward(x []float32, weights [][]float32, B, T, D, N int) []float32 {
	kernel := cpuS4DKernel(weights[:6], T, D, N)
	fftLen := 1
	for fftLen < 2*T {
		fftLen <<= 1
	}
	out := make([]float32, B*T*D)
	for batch := 0; batch < B; batch++ {
		for feature := 0; feature < D; feature++ {
			xFrequency := make([]complex128, fftLen)
			kernelFrequency := make([]complex128, fftLen)
			for frequency := 0; frequency < fftLen; frequency++ {
				for position := 0; position < T; position++ {
					angle := -2 * math.Pi * float64(frequency*position) / float64(fftLen)
					phase := cmplx.Exp(complex(0, angle))
					xFrequency[frequency] += complex(float64(x[(batch*T+position)*D+feature]), 0) * phase
					kernelFrequency[frequency] += complex(float64(kernel[feature*T+position]), 0) * phase
				}
			}
			for frequency := range xFrequency {
				absoluteFrequency := frequency
				if absoluteFrequency > fftLen/2 {
					absoluteFrequency = fftLen - absoluteFrequency
				}
				filter := math.Pow(
					1+float64(absoluteFrequency)/float64(fftLen),
					float64(weights[6][feature]),
				)
				xFrequency[frequency] *= kernelFrequency[frequency] * complex(filter, 0)
			}
			for position := 0; position < T; position++ {
				var value complex128
				for frequency := 0; frequency < fftLen; frequency++ {
					angle := 2 * math.Pi * float64(frequency*position) / float64(fftLen)
					value += xFrequency[frequency] * cmplx.Exp(complex(0, angle))
				}
				index := (batch*T+position)*D + feature
				out[index] = float32(real(value)/float64(fftLen)) + x[index]*weights[5][feature]
			}
		}
	}
	return out
}

func finiteDifferenceS4DSobolevGrads(
	x []float32,
	weights [][]float32,
	B, T, D, N int,
) [][]float32 {
	const eps = float32(1e-3)
	grads := make([][]float32, len(weights))
	for weightIndex := range weights {
		grads[weightIndex] = make([]float32, len(weights[weightIndex]))
		for valueIndex := range weights[weightIndex] {
			original := weights[weightIndex][valueIndex]
			weights[weightIndex][valueIndex] = original + eps
			plus := meanSquareFloat32(cpuS4DSobolevForward(x, weights, B, T, D, N))
			weights[weightIndex][valueIndex] = original - eps
			minus := meanSquareFloat32(cpuS4DSobolevForward(x, weights, B, T, D, N))
			weights[weightIndex][valueIndex] = original
			grads[weightIndex][valueIndex] = float32((plus - minus) / float64(2*eps))
		}
	}
	return grads
}
