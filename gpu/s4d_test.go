//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"math/cmplx"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestS4DKernelMatchesOfficialMinimalReferenceFixture(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 1
		T = 6
		D = 2
		N = 4
	)
	logDT := []float32{float32(math.Log(0.01)), float32(math.Log(0.05))}
	logAReal := repeatFloat32(D*(N/2), float32(math.Log(0.5)))
	aImag := []float32{0, math.Pi, 0, math.Pi}
	cReal := []float32{1, -0.25, 0.3, 0.8}
	cImag := []float32{0.5, 0.75, -0.2, 0.1}
	direct := []float32{0, 0}
	x := make([]float32, B*T*D)

	_, kernel := runS4DInputs(t, 0, x, logDT, logAReal, aImag, cReal, cImag, direct, B, T, D, N)
	// Generated from state-spaces/s4 models/s4/s4d.py S4DKernel:
	// A=-exp(log_A_real)+i*A_imag; C*=expm1(dt*A)/A;
	// K=2*Re(sum(C*exp(dt*A*l))).
	want := []float32{
		0.01472856555, 0.01419255194, 0.01366707466, 0.01315246070, 0.01264902339, 0.01215706213,
		0.10754305633, 0.10151069405, 0.09396679812, 0.08515372111, 0.07533367249, 0.06478128288,
	}
	if diff := maxAbsDiffFloat32(kernel, want); diff > 2e-6 {
		t.Fatalf("S4D reference kernel L_inf=%g want <=2e-6\ngot=%v\nwant=%v", diff, kernel, want)
	}
}

func TestS4DFFTMatchesRecurrentForwardAndGradient(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B = 2
		T = 17
		D = 3
		N = 6
	)
	x, weights := s4dParityFixture(B, T, D, N)
	fftOutput, fftKernel := runS4DWeights(t, 0, x, weights, B, T, D, N)
	scanOutput, scanKernel := runS4DWeights(t, 1, x, weights, B, T, D, N)
	cpuKernel := cpuS4DKernel(weights, T, D, N)
	cpuOutput := cpuS4DForward(x, cpuKernel, weights[5], B, T, D)
	if diff := maxAbsDiffFloat32(fftKernel, cpuKernel); diff > 2e-5 {
		t.Fatalf("MLX/CPU kernel L_inf=%g want <=2e-5", diff)
	}
	if diff := maxAbsDiffFloat32(fftOutput, cpuOutput); diff > 3e-5 {
		t.Fatalf("MLX/CPU forward L_inf=%g want <=3e-5", diff)
	}
	if diff := maxAbsDiffFloat32(fftKernel, scanKernel); diff != 0 {
		t.Fatalf("shared kernels differ by %g", diff)
	}
	if diff := maxAbsDiffFloat32(fftOutput, scanOutput); diff > 2e-5 {
		t.Fatalf("FFT/scan forward L_inf=%g want <=2e-5", diff)
	}

	fftTrainer := createS4DGradTrainer(t, 0, weights, B, T, D, N)
	scanTrainer := createS4DGradTrainer(t, 1, weights, B, T, D, N)
	inputs := []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}
	fftLoss, err := TrainerComputeMeanSquareGrads(fftTrainer, inputs, "output")
	if err != nil {
		t.Fatalf("FFT gradients: %v", err)
	}
	scanLoss, err := TrainerComputeMeanSquareGrads(scanTrainer, inputs, "output")
	if err != nil {
		t.Fatalf("scan gradients: %v", err)
	}
	if diff := math.Abs(float64(fftLoss - scanLoss)); diff > 1e-5 {
		t.Fatalf("FFT/scan mean-square loss diff=%g", diff)
	}
	fftGrads := make([][]float32, len(weights))
	for wi, weight := range weights {
		fftGrad := make([]float32, len(weight))
		scanGrad := make([]float32, len(weight))
		if err := TrainerReadGrad(fftTrainer, wi, fftGrad); err != nil {
			t.Fatalf("read FFT grad %d: %v", wi, err)
		}
		if err := TrainerReadGrad(scanTrainer, wi, scanGrad); err != nil {
			t.Fatalf("read scan grad %d: %v", wi, err)
		}
		for _, value := range fftGrad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("FFT grad %d contains non-finite value", wi)
			}
		}
		if diff := maxAbsDiffFloat32(fftGrad, scanGrad); diff > 5e-4 {
			t.Fatalf("FFT/scan grad %d L_inf=%g want <=5e-4", wi, diff)
		}
		fftGrads[wi] = fftGrad
	}
	cpuGrads := finiteDifferenceS4DGrads(x, weights, B, T, D, N)
	for wi := range weights {
		for i := range weights[wi] {
			got := fftGrads[wi][i]
			want := cpuGrads[wi][i]
			tolerance := float32(3e-3 + 0.05*math.Abs(float64(want)))
			if diff := float32(math.Abs(float64(got - want))); diff > tolerance {
				t.Fatalf("MLX/CPU grad w%d[%d]=%g want=%g diff=%g tolerance=%g", wi, i, got, want, diff, tolerance)
			}
		}
	}
}

func BenchmarkS4DFFTLongSequence(b *testing.B) {
	if !Available() {
		b.Skip("MLX backend not available")
	}
	const (
		B = 1
		T = 4096
		D = 64
		N = 64
	)
	x, weights := s4dParityFixture(B, T, D, N)
	prog := lowerS4DProgram(b, 0, B, T, D, N, false)
	defer prog.Destroy()
	handles := s4dWeightHandles(b, weights, D, N)
	defer FreeHandles(handles)
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := EvalProgramOutput(prog, handles, inputs, "output"); err != nil {
			b.Fatal(err)
		}
	}
}

func runS4DInputs(
	t *testing.T,
	mode int,
	x, logDT, logAReal, aImag, cReal, cImag, direct []float32,
	B, T, D, N int,
) ([]float32, []float32) {
	t.Helper()
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareInput("log_dt", ir.TensorFloat32, []int{D})
	prog.DeclareInput("log_a_real", ir.TensorFloat32, []int{D, N / 2})
	prog.DeclareInput("a_imag", ir.TensorFloat32, []int{D, N / 2})
	prog.DeclareInput("c_real", ir.TensorFloat32, []int{D, N / 2})
	prog.DeclareInput("c_imag", ir.TensorFloat32, []int{D, N / 2})
	prog.DeclareInput("direct", ir.TensorFloat32, []int{D})
	prog.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("kernel", ir.TensorFloat32, []int{D, T})
	prog.S4D("x", "log_dt", "log_a_real", "a_imag", "c_real", "c_imag", "direct", "output", "kernel", B, T, D, N, mode)
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandle(dummy)
	inputs := []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x},
		{Name: "log_dt", DType: TensorFloat32, Shape: []int{D}, Data: logDT},
		{Name: "log_a_real", DType: TensorFloat32, Shape: []int{D, N / 2}, Data: logAReal},
		{Name: "a_imag", DType: TensorFloat32, Shape: []int{D, N / 2}, Data: aImag},
		{Name: "c_real", DType: TensorFloat32, Shape: []int{D, N / 2}, Data: cReal},
		{Name: "c_imag", DType: TensorFloat32, Shape: []int{D, N / 2}, Data: cImag},
		{Name: "direct", DType: TensorFloat32, Shape: []int{D}, Data: direct},
	}
	output, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "output")
	if err != nil {
		t.Fatalf("EvalProgramOutput(output): %v", err)
	}
	kernel, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "kernel")
	if err != nil {
		t.Fatalf("EvalProgramOutput(kernel): %v", err)
	}
	return output, kernel
}

func runS4DWeights(
	t *testing.T,
	mode int,
	x []float32,
	weights [][]float32,
	B, T, D, N int,
) ([]float32, []float32) {
	t.Helper()
	prog := lowerS4DProgram(t, mode, B, T, D, N, true)
	defer prog.Destroy()
	handles := s4dWeightHandles(t, weights, D, N)
	defer FreeHandles(handles)
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x}}
	output, err := EvalProgramOutput(prog, handles, inputs, "output")
	if err != nil {
		t.Fatal(err)
	}
	kernel, err := EvalProgramOutput(prog, handles, inputs, "kernel")
	if err != nil {
		t.Fatal(err)
	}
	return output, kernel
}

type s4dTestFataler interface {
	Helper()
	Fatal(args ...any)
}

func lowerS4DProgram(tb s4dTestFataler, mode, B, T, D, N int, declareKernel bool) *Program {
	tb.Helper()
	prog := ir.NewProgram(6)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	if declareKernel {
		prog.DeclareOutput("kernel", ir.TensorFloat32, []int{D, T})
	}
	prog.S4D("x", "w0", "w1", "w2", "w3", "w4", "w5", "output", "kernel", B, T, D, N, mode)
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		tb.Fatal(err)
	}
	return gpuProg
}

func s4dParityFixture(B, T, D, N int) ([]float32, [][]float32) {
	x := make([]float32, B*T*D)
	for i := range x {
		x[i] = 0.15 * float32(math.Sin(float64(i+1)*0.37))
	}
	pairs := N / 2
	weights := make([][]float32, 6)
	weights[0] = make([]float32, D)
	weights[1] = repeatFloat32(D*pairs, float32(math.Log(0.5)))
	weights[2] = make([]float32, D*pairs)
	weights[3] = make([]float32, D*pairs)
	weights[4] = make([]float32, D*pairs)
	weights[5] = make([]float32, D)
	for d := 0; d < D; d++ {
		weights[0][d] = float32(math.Log(0.003 + 0.007*float64(d+1)/float64(D)))
		weights[5][d] = 0.1 * float32(d+1)
		for n := 0; n < pairs; n++ {
			i := d*pairs + n
			weights[2][i] = math.Pi * float32(n)
			weights[3][i] = 0.2 * float32(math.Sin(float64(i+1)))
			weights[4][i] = 0.2 * float32(math.Cos(float64(i+1)))
		}
	}
	return x, weights
}

func s4dWeightHandles(tb s4dTestFataler, weights [][]float32, D, N int) []int64 {
	tb.Helper()
	shapes := [][]int{{D}, {D, N / 2}, {D, N / 2}, {D, N / 2}, {D, N / 2}, {D}}
	handles := make([]int64, len(weights))
	for i, data := range weights {
		handle, err := FromDataShape(append([]float32(nil), data...), shapes[i])
		if err != nil {
			FreeHandles(handles[:i])
			tb.Fatal(err)
		}
		handles[i] = handle
	}
	return handles
}

func createS4DGradTrainer(t *testing.T, mode int, weights [][]float32, B, T, D, N int) TrainerHandle {
	t.Helper()
	prog := lowerS4DProgram(t, mode, B, T, D, N, false)
	handles := s4dWeightHandles(t, weights, D, N)
	optimizerWeights := make([]WeightOptimizer, len(weights))
	for i := range optimizerWeights {
		optimizerWeights[i] = WeightOptimizer{GroupIndex: 0}
	}
	trainer, err := CreateTrainer(prog, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind: OptimizerAdamW, LR: 0, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8,
		}},
		Weights:       optimizerWeights,
		DefaultBaseLR: 0,
	})
	if err != nil {
		prog.Destroy()
		FreeHandles(handles)
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = TrainerFlush(trainer)
		TrainerDestroy(trainer)
		FreeHandles(handles)
		prog.Destroy()
	})
	return trainer
}

func repeatFloat32(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func cpuS4DKernel(weights [][]float32, T, D, N int) []float32 {
	pairs := N / 2
	kernel := make([]float32, D*T)
	for d := 0; d < D; d++ {
		dt := math.Exp(float64(weights[0][d]))
		for n := 0; n < pairs; n++ {
			i := d*pairs + n
			a := complex(-math.Exp(float64(weights[1][i])), float64(weights[2][i]))
			abar := cmplx.Exp(complex(dt, 0) * a)
			bbar := (abar - 1) / a
			c := complex(float64(weights[3][i]), float64(weights[4][i]))
			gamma := c * bbar
			power := complex(1, 0)
			for pos := 0; pos < T; pos++ {
				kernel[d*T+pos] += float32(2 * real(gamma*power))
				power *= abar
			}
		}
	}
	return kernel
}

func cpuS4DForward(x, kernel, direct []float32, B, T, D int) []float32 {
	out := make([]float32, len(x))
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				value := float64(x[(b*T+t)*D+d] * direct[d])
				for lag := 0; lag <= t; lag++ {
					value += float64(kernel[d*T+lag] * x[(b*T+t-lag)*D+d])
				}
				out[(b*T+t)*D+d] = float32(value)
			}
		}
	}
	return out
}

func finiteDifferenceS4DGrads(x []float32, weights [][]float32, B, T, D, N int) [][]float32 {
	const eps = float32(1e-3)
	grads := make([][]float32, len(weights))
	for wi := range weights {
		grads[wi] = make([]float32, len(weights[wi]))
		for i := range weights[wi] {
			original := weights[wi][i]
			weights[wi][i] = original + eps
			plus := meanSquareFloat32(cpuS4DForward(x, cpuS4DKernel(weights, T, D, N), weights[5], B, T, D))
			weights[wi][i] = original - eps
			minus := meanSquareFloat32(cpuS4DForward(x, cpuS4DKernel(weights, T, D, N), weights[5], B, T, D))
			weights[wi][i] = original
			grads[wi][i] = float32((plus - minus) / float64(2*eps))
		}
	}
	return grads
}

func meanSquareFloat32(values []float32) float64 {
	var total float64
	for _, value := range values {
		total += float64(value) * float64(value)
	}
	return total / float64(len(values))
}
