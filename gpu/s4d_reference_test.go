//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"runtime"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestS4DReferenceBidirectionalGroupedBilinearForward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B    = 1
		T    = 5
		D    = 4
		N    = 4
		nSSM = 2
	)
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	prog := lowerS4DReferenceProgram(t, B, T, D, N, nSSM, true)
	defer prog.Destroy()
	handles := s4dReferenceWeightHandles(t, weights, D, N, nSSM)
	defer FreeHandles(handles)
	inputs := []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}

	got, err := EvalProgramOutput(prog, handles, inputs, "output")
	if err != nil {
		t.Fatal(err)
	}
	gotKernel, err := EvalProgramOutput(prog, handles, inputs, "kernel")
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(gotKernel, s4dOfficialAdvancedKernelFixture); diff > 3e-5 {
		t.Fatalf("pinned reference kernel L_inf=%g want <=3e-5", diff)
	}
	if diff := maxAbsDiffFloat32(got, s4dOfficialAdvancedOutputFixture); diff > 4e-5 {
		t.Fatalf("pinned reference output L_inf=%g want <=4e-5", diff)
	}
	want, wantKernel := cpuS4DReferenceForward(x, weights, B, T, D, N, nSSM, true)
	if diff := maxAbsDiffFloat32(gotKernel, wantKernel); diff > 3e-5 {
		t.Fatalf("reference bidirectional kernel L_inf=%g want <=3e-5\ngot=%v\nwant=%v", diff, gotKernel, wantKernel)
	}
	if diff := maxAbsDiffFloat32(got, want); diff > 4e-5 {
		t.Fatalf("reference bidirectional output L_inf=%g want <=4e-5\ngot=%v\nwant=%v", diff, got, want)
	}
}

func TestS4DReferenceBidirectionalGroupedBilinearBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N, nSSM = 1, 5, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	gradients := s4dReferenceGradients(t, x, weights, B, T, D, N, nSSM)
	if len(gradients) != len(s4dOfficialAdvancedGradientFixtures) {
		t.Fatalf("gradient tensor count=%d want=%d", len(gradients), len(s4dOfficialAdvancedGradientFixtures))
	}
	// MLX and PyTorch use independent float32 FFT and complex-autodiff reduction
	// orders. The mixed tolerance remains over four orders of magnitude tighter
	// in absolute terms than the former one-coordinate finite-difference check.
	const absoluteTolerance = float32(3e-7)
	const relativeTolerance = float32(0.01)
	for wi, gradient := range gradients {
		want := s4dOfficialAdvancedGradientFixtures[wi]
		if len(gradient) != len(want) {
			t.Fatalf("gradient w%d length=%d want=%d", wi, len(gradient), len(want))
		}
		for i, value := range gradient {
			tolerance := absoluteTolerance + relativeTolerance*float32(math.Abs(float64(want[i])))
			if diff := float32(math.Abs(float64(value - want[i]))); diff > tolerance {
				t.Fatalf("gradient w%d[%d]=%g reference=%g diff=%g tolerance=%g", wi, i, value, want[i], diff, tolerance)
			}
		}
	}
}

func TestS4DBidirectionalMetalKernelMatchesMLXFallbackForwardAndBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	if runtime.GOOS != "darwin" {
		t.Skip("native S4D kernel primitive is Metal-only")
	}
	const B, T, D, N, nSSM = 1, 7, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	primitiveOutput, primitiveKernel := runS4DReferenceWeights(
		t, x, weights, B, T, D, N, nSSM, true,
	)
	primitiveGradients := s4dReferenceGradients(t, x, weights, B, T, D, N, nSSM)

	t.Setenv("MIXLAB_S4D_DISABLE_METAL_KERNEL_PRIMITIVE", "1")
	fallbackOutput, fallbackKernel := runS4DReferenceWeights(
		t, x, weights, B, T, D, N, nSSM, true,
	)
	fallbackGradients := s4dReferenceGradients(t, x, weights, B, T, D, N, nSSM)

	if diff := maxAbsDiffFloat32(primitiveKernel, fallbackKernel); diff > 3e-5 {
		t.Fatalf("Metal/fallback kernel L_inf=%g want <=3e-5", diff)
	}
	if diff := maxAbsDiffFloat32(primitiveOutput, fallbackOutput); diff > 4e-5 {
		t.Fatalf("Metal/fallback output L_inf=%g want <=4e-5", diff)
	}
	for weightIndex := range primitiveGradients {
		if diff := maxAbsDiffFloat32(
			primitiveGradients[weightIndex], fallbackGradients[weightIndex],
		); diff > 2e-5 {
			t.Fatalf("Metal/fallback gradient w%d L_inf=%g want <=2e-5", weightIndex, diff)
		}
	}
}

func TestS4DReferenceBackwardFiniteDifferenceSelfConsistency(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D, N, nSSM = 1, 5, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	gradients := s4dReferenceGradients(t, x, weights, B, T, D, N, nSSM)
	for wi, gradient := range gradients {
		for i, value := range gradient {
			want := finiteDifferenceS4DReferenceWeight(x, weights, wi, i, B, T, D, N, nSSM)
			tolerance := float32(5e-6 + 0.03*math.Abs(float64(want)))
			if diff := float32(math.Abs(float64(value - want))); diff > tolerance {
				t.Fatalf("self-consistency gradient w%d[%d]=%g finite_difference=%g diff=%g tolerance=%g", wi, i, value, want, diff, tolerance)
			}
		}
	}
}

func s4dReferenceGradients(
	t *testing.T,
	x []float32,
	weights [][]float32,
	B, T, D, N, nSSM int,
) [][]float32 {
	t.Helper()
	prog := lowerS4DReferenceProgram(t, B, T, D, N, nSSM, true)
	defer prog.Destroy()
	handles := s4dReferenceWeightHandles(t, weights, D, N, nSSM)
	defer FreeHandles(handles)
	optimizerWeights := make([]WeightOptimizer, len(weights))
	for i := range optimizerWeights {
		optimizerWeights[i] = WeightOptimizer{GroupIndex: 0}
	}
	trainer, err := CreateTrainer(prog, handles, TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind: OptimizerAdamW, LR: 0, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8,
		}},
		Weights: optimizerWeights, DefaultBaseLR: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	inputs := []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}
	if _, err := TrainerComputeMeanSquareGrads(trainer, inputs, "output"); err != nil {
		t.Fatal(err)
	}
	gradients := make([][]float32, len(weights))
	for wi, weight := range weights {
		gradients[wi] = make([]float32, len(weight))
		if err := TrainerReadGrad(trainer, wi, gradients[wi]); err != nil {
			t.Fatalf("read gradient %d: %v", wi, err)
		}
		for _, value := range gradients[wi] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("gradient %d contains non-finite value", wi)
			}
		}
	}
	return gradients
}

func TestS4DReferenceZeroBackwardRecoversUnidirectional(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B    = 1
		T    = 5
		D    = 4
		N    = 4
		nSSM = 2
	)
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	for i := range weights[7] {
		weights[7][i] = 0
		weights[8][i] = 0
	}
	bidirectional, _ := runS4DReferenceWeights(t, x, weights, B, T, D, N, nSSM, true)
	unidirectionalWeights := append([][]float32(nil), weights[:7]...)
	unidirectionalWeights = append(unidirectionalWeights, weights[9])
	unidirectional, _ := runS4DReferenceWeights(t, x, unidirectionalWeights, B, T, D, N, nSSM, false)
	if diff := maxAbsDiffFloat32(bidirectional, unidirectional); diff > 3e-5 {
		t.Fatalf("zero backward/unidirectional L_inf=%g want <=3e-5", diff)
	}
}

func lowerS4DReferenceProgram(
	tb s4dTestFataler,
	B, T, D, N, nSSM int,
	bidirectional bool,
) *Program {
	tb.Helper()
	weightCount := 8
	if bidirectional {
		weightCount = 10
	}
	prog := ir.NewProgram(weightCount)
	prog.DeclareInput("x", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("output", ir.TensorFloat32, []int{B * T, D})
	kernelT := T
	if bidirectional {
		kernelT = 2 * T
	}
	prog.DeclareOutput("kernel", ir.TensorFloat32, []int{D, kernelT})
	inputNames := []string{"x", "w0", "w1", "w2", "w3", "w4", "w5", "w6"}
	inputNames = append(inputNames, "w7")
	if bidirectional {
		inputNames = []string{"x", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"}
	}
	prog.S4DAdvanced(
		inputNames,
		"output",
		"kernel",
		B,
		T,
		D,
		N,
		nSSM,
		bidirectional,
		ir.S4DDiscretizationBilinear,
		true,
	)
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		tb.Fatal(err)
	}
	return gpuProg
}

func runS4DReferenceWeights(
	t *testing.T,
	x []float32,
	weights [][]float32,
	B, T, D, N, nSSM int,
	bidirectional bool,
) ([]float32, []float32) {
	t.Helper()
	prog := lowerS4DReferenceProgram(t, B, T, D, N, nSSM, bidirectional)
	defer prog.Destroy()
	handles := s4dReferenceWeightHandles(t, weights, D, N, nSSM)
	defer FreeHandles(handles)
	inputs := []TensorInput{{
		Name: "x", DType: TensorFloat32, Shape: []int{B * T, D}, Data: x,
	}}
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

func s4dReferenceFixture(B, T, D, N, nSSM int) ([]float32, [][]float32) {
	pairs := N / 2
	x := make([]float32, B*T*D)
	for i := range x {
		x[i] = 0.2 * float32(math.Sin(float64(i+1)*0.31))
	}
	weights := [][]float32{
		make([]float32, D),
		make([]float32, nSSM*pairs),
		make([]float32, nSSM*pairs),
		make([]float32, nSSM*pairs),
		make([]float32, nSSM*pairs),
		make([]float32, D*pairs),
		make([]float32, D*pairs),
		make([]float32, D*pairs),
		make([]float32, D*pairs),
		make([]float32, D),
	}
	for d := 0; d < D; d++ {
		weights[0][d] = float32(math.Log(0.004 + 0.002*float64(d)))
		weights[9][d] = 0.03 * float32(d+1)
		for n := 0; n < pairs; n++ {
			i := d*pairs + n
			weights[5][i] = 0.15 * float32(math.Sin(float64(i+1)))
			weights[6][i] = 0.12 * float32(math.Cos(float64(i+1)))
			weights[7][i] = 0.11 * float32(math.Cos(float64(i+2)))
			weights[8][i] = 0.09 * float32(math.Sin(float64(i+2)))
		}
	}
	for g := 0; g < nSSM; g++ {
		for n := 0; n < pairs; n++ {
			i := g*pairs + n
			weights[1][i] = float32(math.Log(0.5 + 0.25*float64(g)))
			weights[2][i] = math.Pi * float32(n+g)
			weights[3][i] = 1 + 0.05*float32(i)
			weights[4][i] = -0.03 * float32(i+1)
		}
	}
	return x, weights
}

// The two halves guard different properties. The per-group row checks keep every
// grouped weight -- log_A_real included -- distinguishable across n_ssm, so a
// mis-indexed group lookup cannot read a neighbour's value and still agree. The
// mapping comparison keeps the fixture's channel->group assignment observable at
// all; it already held before log_A_real varied by group (kernel L_inf 4.8e-4),
// so it is a floor to defend, not a property the row checks established.
func TestS4DReferenceFixtureBreaksGroupedSymmetry(t *testing.T) {
	const B, T, D, N, nSSM = 1, 5, 4, 4, 2
	x, weights := s4dReferenceFixture(B, T, D, N, nSSM)
	pairs := N / 2

	if equalFloat32Slices(weights[0][:D/2], weights[0][D/2:]) {
		t.Fatal("log_dt fixture must vary across channel groups")
	}
	for _, tc := range []struct {
		name  string
		index int
	}{
		{name: "log_A_real", index: 1},
		{name: "A_imag", index: 2},
		{name: "B_real", index: 3},
		{name: "B_imag", index: 4},
	} {
		if equalFloat32Slices(weights[tc.index][:pairs], weights[tc.index][pairs:2*pairs]) {
			t.Fatalf("%s fixture rows must differ across n_ssm groups", tc.name)
		}
	}

	interleavedOutput, interleavedKernel := cpuS4DReferenceForward(
		x, weights, B, T, D, N, nSSM, true,
	)
	blockOutput, blockKernel := cpuS4DReferenceForwardWithGroupMapping(
		x, weights, B, T, D, N, nSSM, true,
		func(channel int) int { return channel / (D / nSSM) },
	)
	if diff := maxAbsDiffFloat32(interleavedKernel, blockKernel); diff <= 3e-5 {
		t.Fatalf("fixture does not distinguish grouped kernel mappings: L_inf=%g", diff)
	}
	if diff := maxAbsDiffFloat32(interleavedOutput, blockOutput); diff <= 4e-5 {
		t.Fatalf("fixture does not distinguish grouped output mappings: L_inf=%g", diff)
	}
}

func equalFloat32Slices(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func s4dReferenceWeightHandles(
	tb s4dTestFataler,
	weights [][]float32,
	D, N, nSSM int,
) []int64 {
	tb.Helper()
	pairs := N / 2
	shapes := [][]int{
		{D},
		{nSSM, pairs},
		{nSSM, pairs},
		{nSSM, pairs},
		{nSSM, pairs},
		{D, pairs},
		{D, pairs},
	}
	if len(weights) == 10 {
		shapes = append(shapes, []int{D, pairs}, []int{D, pairs}, []int{D})
	} else {
		shapes = append(shapes, []int{D})
	}
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

func cpuS4DReferenceForward(
	x []float32,
	weights [][]float32,
	B, T, D, N, nSSM int,
	bidirectional bool,
) ([]float32, []float32) {
	return cpuS4DReferenceForwardWithGroupMapping(
		x,
		weights,
		B,
		T,
		D,
		N,
		nSSM,
		bidirectional,
		func(channel int) int { return channel % nSSM },
	)
}

func cpuS4DReferenceForwardWithGroupMapping(
	x []float32,
	weights [][]float32,
	B, T, D, N, nSSM int,
	bidirectional bool,
	groupForChannel func(int) int,
) ([]float32, []float32) {
	pairs := N / 2
	kernelT := T
	if bidirectional {
		kernelT = 2 * T
	}
	kernel := make([]float32, D*kernelT)
	for d := 0; d < D; d++ {
		group := groupForChannel(d)
		dt := math.Exp(float64(weights[0][d]))
		for n := 0; n < pairs; n++ {
			stateIdx := group*pairs + n
			a := complex(
				-math.Exp(float64(weights[1][stateIdx])),
				float64(weights[2][stateIdx]),
			)
			b := complex(float64(weights[3][stateIdx]), float64(weights[4][stateIdx]))
			abar := (1 + complex(0.5*dt, 0)*a) / (1 - complex(0.5*dt, 0)*a)
			bbar := complex(dt, 0) * b / (1 - complex(0.5*dt, 0)*a)
			cForward := complex(
				float64(weights[5][d*pairs+n]),
				float64(weights[6][d*pairs+n]),
			)
			power := complex(1, 0)
			for pos := 0; pos < T; pos++ {
				kernel[d*kernelT+pos] += float32(2 * real(cForward*bbar*power))
				power *= abar
			}
			if bidirectional {
				cBackward := complex(
					float64(weights[7][d*pairs+n]),
					float64(weights[8][d*pairs+n]),
				)
				power = 1
				for pos := 0; pos < T; pos++ {
					reversedPos := 2*T - 1 - pos
					kernel[d*kernelT+reversedPos] += float32(2 * real(cBackward*bbar*power))
					power *= abar
				}
			}
		}
	}
	direct := weights[len(weights)-1]
	out := make([]float32, len(x))
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			for d := 0; d < D; d++ {
				value := float64(x[(b*T+t)*D+d] * direct[d])
				for source := 0; source < T; source++ {
					lag := t - source
					if lag < 0 {
						lag += kernelT
					}
					value += float64(x[(b*T+source)*D+d] * kernel[d*kernelT+lag])
				}
				out[(b*T+t)*D+d] = float32(value)
			}
		}
	}
	return out, kernel
}

func finiteDifferenceS4DReferenceWeight(
	x []float32,
	weights [][]float32,
	weightIndex, valueIndex int,
	B, T, D, N, nSSM int,
) float32 {
	const eps = float32(1e-3)
	original := weights[weightIndex][valueIndex]
	weights[weightIndex][valueIndex] = original + eps
	plusOut, _ := cpuS4DReferenceForward(x, weights, B, T, D, N, nSSM, true)
	plus := meanSquareFloat32(plusOut)
	weights[weightIndex][valueIndex] = original - eps
	minusOut, _ := cpuS4DReferenceForward(x, weights, B, T, D, N, nSSM, true)
	minus := meanSquareFloat32(minusOut)
	weights[weightIndex][valueIndex] = original
	return float32((plus - minus) / float64(2*eps))
}
