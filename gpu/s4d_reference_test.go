//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestS4DReferenceBidirectionalGroupedBilinearForwardAndBackward(t *testing.T) {
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
	// Generated from the pinned state-spaces/s4 v3.0.0 bilinear S4D
	// construction. The backward half uses the reference's one-position offset.
	fixtureKernel := []float32{
		0.002147922309, 0.002149529379, 0.002150943803, 0.0021521658, 0.00215319562,
		-0.001251425183, -0.001252519065, -0.001253468585, -0.001254273121, -0.00125493207,
		-0.001265023278, -0.001243701524, -0.001221981124, -0.001199872575, -0.001177386449,
		-0.0004572928295, -0.0004788437295, -0.0005006622494, -0.0005227429677, -0.0005450803376,
		-0.00309085857, -0.003205462166, -0.003315747596, -0.003421483273, -0.003522447824,
		0.003208791609, 0.003272809461, 0.00333249285, 0.00338764719, 0.003438085555,
		0.005679525575, 0.005626399379, 0.005558132086, 0.005474932666, 0.005377062697,
		-0.002837327262, -0.00276004643, -0.002672173386, -0.002573777497, -0.002464968007,
	}
	fixtureOutput := []float32{
		0.002031188804, 0.006871779731, 0.01340578, 0.02431566743,
		0.006880301958, 0.01127223151, 0.01272581847, 0.01732010161,
		0.003182990792, 0.0002771852203, -0.006616455184, -0.01147814445,
		-0.004067369576, -0.01125539348, -0.01853760893, -0.02322492008,
		-0.005079413711, -0.007741502018, -0.006973650025, -0.002097363671,
	}
	if diff := maxAbsDiffFloat32(gotKernel, fixtureKernel); diff > 3e-5 {
		t.Fatalf("pinned reference kernel L_inf=%g want <=3e-5", diff)
	}
	if diff := maxAbsDiffFloat32(got, fixtureOutput); diff > 4e-5 {
		t.Fatalf("pinned reference output L_inf=%g want <=4e-5", diff)
	}
	want, wantKernel := cpuS4DReferenceForward(x, weights, B, T, D, N, nSSM, true)
	if diff := maxAbsDiffFloat32(gotKernel, wantKernel); diff > 3e-5 {
		t.Fatalf("reference bidirectional kernel L_inf=%g want <=3e-5\ngot=%v\nwant=%v", diff, gotKernel, wantKernel)
	}
	if diff := maxAbsDiffFloat32(got, want); diff > 4e-5 {
		t.Fatalf("reference bidirectional output L_inf=%g want <=4e-5\ngot=%v\nwant=%v", diff, got, want)
	}

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
	if _, err := TrainerComputeMeanSquareGrads(trainer, inputs, "output"); err != nil {
		t.Fatal(err)
	}
	for wi := range weights {
		grad := make([]float32, len(weights[wi]))
		if err := TrainerReadGrad(trainer, wi, grad); err != nil {
			t.Fatalf("read gradient %d: %v", wi, err)
		}
		for _, value := range grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("gradient %d contains non-finite value", wi)
			}
		}
		wantGrad := finiteDifferenceS4DReferenceWeight(x, weights, wi, 0, B, T, D, N, nSSM)
		tolerance := float32(5e-3 + 0.06*math.Abs(float64(wantGrad)))
		if diff := float32(math.Abs(float64(grad[0] - wantGrad))); diff > tolerance {
			t.Fatalf("gradient w%d[0]=%g want=%g diff=%g tolerance=%g", wi, grad[0], wantGrad, diff, tolerance)
		}
	}
}

func TestS4DReferenceZeroBackwardRecoversUnidirectional(t *testing.T) {
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
		repeatFloat32(nSSM*pairs, float32(math.Log(0.5))),
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
			weights[2][i] = math.Pi * float32(n+g)
			weights[3][i] = 1 + 0.05*float32(i)
			weights[4][i] = -0.03 * float32(i+1)
		}
	}
	return x, weights
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
	pairs := N / 2
	kernelT := T
	if bidirectional {
		kernelT = 2 * T
	}
	kernel := make([]float32, D*kernelT)
	for d := 0; d < D; d++ {
		group := d / (D / nSSM)
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
