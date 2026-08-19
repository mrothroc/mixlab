//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

const (
	codebookRefB = 2
	codebookRefT = 4
	codebookRefQ = 2
	codebookRefV = 8
	codebookRefD = 6
	codebookRefH = 6
)

func TestDiscreteCodebooksDASBReferenceForwardAndBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := lowerCodebookReferenceProgram(t, true)
	defer prog.Destroy()
	handles := codebookReferenceWeightHandles(t)
	defer FreeHandles(handles)
	inputs := []TensorInput{{
		Name: "codebook_tokens", DType: TensorInt32,
		Shape: []int{codebookRefB, codebookRefT, codebookRefQ}, Data: codebookReferenceTokens,
	}}
	got, err := EvalProgramOutput(prog, handles, inputs, "output")
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(got, codebookReferenceOutput); diff > 2e-6 {
		t.Fatalf("pinned DASB reference output L_inf=%g want <=2e-6\ngot=%v\nwant=%v", diff, got, codebookReferenceOutput)
	}

	optimizers := make([]WeightOptimizer, len(handles))
	for i := range optimizers {
		optimizers[i] = WeightOptimizer{GroupIndex: 0}
	}
	trainer, err := CreateTrainer(prog, handles, TrainerOptimizerSpec{
		Groups:  []OptimizerGroup{{Kind: OptimizerAdamW, LR: 0, Beta1: 0.9, Beta2: 0.95, Epsilon: 1e-8}},
		Weights: optimizers, DefaultBaseLR: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	if _, err := TrainerComputeMeanSquareGrads(trainer, inputs, "output"); err != nil {
		t.Fatal(err)
	}
	for wi, want := range codebookReferenceGradients {
		gotGradient := make([]float32, len(want))
		if err := TrainerReadGrad(trainer, wi, gotGradient); err != nil {
			t.Fatalf("read gradient %d: %v", wi, err)
		}
		for i, value := range gotGradient {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("gradient w%d[%d] is non-finite: %g", wi, i, value)
			}
			tolerance := float32(2e-6 + 2e-4*math.Abs(float64(want[i])))
			if diff := float32(math.Abs(float64(value - want[i]))); diff > tolerance {
				t.Fatalf("gradient w%d[%d]=%g reference=%g diff=%g tolerance=%g", wi, i, value, want[i], diff, tolerance)
			}
		}
	}
}

func TestDiscreteCodebooksMeanFusionHandOracle(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := lowerCodebookReferenceProgram(t, false)
	defer prog.Destroy()
	embedding := make([]float32, codebookRefQ*codebookRefV*codebookRefD)
	for row := 0; row < codebookRefQ*codebookRefV; row++ {
		for d := 0; d < codebookRefD; d++ {
			embedding[row*codebookRefD+d] = float32(row*10 + d)
		}
	}
	handle, err := FromDataShape(embedding, []int{codebookRefQ * codebookRefV, codebookRefD})
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandles([]int64{handle})
	inputs := []TensorInput{{
		Name: "codebook_tokens", DType: TensorInt32,
		Shape: []int{codebookRefB, codebookRefT, codebookRefQ}, Data: codebookReferenceTokens,
	}}
	got, err := EvalProgramOutput(prog, []int64{handle}, inputs, "output")
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, codebookRefB*codebookRefT*codebookRefD)
	for token := 0; token < codebookRefB*codebookRefT; token++ {
		first := int(codebookReferenceTokens[token*codebookRefQ])
		second := codebookRefV + int(codebookReferenceTokens[token*codebookRefQ+1])
		for d := 0; d < codebookRefD; d++ {
			want[token*codebookRefD+d] = (embedding[first*codebookRefD+d] + embedding[second*codebookRefD+d]) / 2
		}
	}
	if diff := maxAbsDiffFloat32(got, want); diff != 0 {
		t.Fatalf("mean codebook fusion L_inf=%g\ngot=%v\nwant=%v", diff, got, want)
	}
}

func lowerCodebookReferenceProgram(t *testing.T, attention bool) *Program {
	t.Helper()
	weightCount := 1
	if attention {
		weightCount = 4
	}
	prog := ir.NewProgram(weightCount)
	prog.DeclareInput("codebook_tokens", ir.TensorInt32, []int{codebookRefB, codebookRefT, codebookRefQ})
	prog.CodebookOffset("codebook_tokens", codebookRefQ, codebookRefV, "indices")
	prog.Embed("w0", "indices", "embeddings")
	if attention {
		prog.Reshape("embeddings", []int{codebookRefB * codebookRefT * codebookRefQ, codebookRefD}, "flat")
		prog.MatMul("flat", "w1", "h0")
		prog.Add("h0", "w2", "h1")
		prog.ReLU("h1", "h")
		prog.MatMul("h", "w3", "scores0")
		prog.Reshape("scores0", []int{codebookRefB * codebookRefT, codebookRefQ}, "scores")
		prog.Softmax("scores", -1, "weights")
		prog.Reshape("weights", []int{codebookRefB * codebookRefT, 1, codebookRefQ}, "weights_b1q")
		prog.Reshape("embeddings", []int{codebookRefB * codebookRefT, codebookRefQ, codebookRefD}, "embeddings_bqd")
		prog.MatMul("weights_b1q", "embeddings_bqd", "fused_b1d")
		prog.Reshape("fused_b1d", []int{codebookRefB, codebookRefT, codebookRefD}, "output")
	} else {
		prog.MeanAxis("embeddings", 2, "output")
	}
	prog.DeclareOutput("output", ir.TensorFloat32, []int{codebookRefB, codebookRefT, codebookRefD})
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	return gpuProg
}

func codebookReferenceWeightHandles(t *testing.T) []int64 {
	t.Helper()
	shapes := [][]int{{codebookRefQ * codebookRefV, codebookRefD}, {codebookRefD, codebookRefH}, {codebookRefH}, {codebookRefH, 1}}
	handles := make([]int64, len(shapes))
	for i, shape := range shapes {
		handle, err := FromDataShape(append([]float32(nil), codebookReferenceWeights[i]...), shape)
		if err != nil {
			FreeHandles(handles[:i])
			t.Fatalf("upload reference weight %d: %v", i, err)
		}
		handles[i] = handle
	}
	return handles
}
