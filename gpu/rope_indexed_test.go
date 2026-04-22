//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

const (
	ropeIndexedTestB       = 1
	ropeIndexedTestT       = 4
	ropeIndexedTestH       = 2
	ropeIndexedTestHeadDim = 4
	ropeIndexedTestD       = ropeIndexedTestH * ropeIndexedTestHeadDim
	ropeIndexedTestVocab   = ropeIndexedTestT
	ropeIndexedTestBase    = 10000.0
	ropeIndexedTestTol     = 1e-5
)

func buildRoPEIndexedTestProgram(indexed bool) *ir.Program {
	prog := ir.NewProgram(2)
	prog.DeclareInput("tokens", ir.TensorInt32, []int{ropeIndexedTestB, ropeIndexedTestT})
	prog.DeclareInput("targets", ir.TensorInt32, []int{ropeIndexedTestB * ropeIndexedTestT})
	if indexed {
		prog.DeclareInput("positions", ir.TensorInt32, []int{ropeIndexedTestT})
	}
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("qr", ir.TensorFloat32, []int{
		ropeIndexedTestB,
		ropeIndexedTestH,
		ropeIndexedTestT,
		ropeIndexedTestHeadDim,
	})

	prog.Embed("w0", "tokens", "emb")
	prog.Reshape("emb", []int{ropeIndexedTestB * ropeIndexedTestT, ropeIndexedTestD}, "flat")
	prog.Reshape("flat", []int{ropeIndexedTestB, ropeIndexedTestT, ropeIndexedTestH, ropeIndexedTestHeadDim}, "q_bthd")
	prog.Transpose("q_bthd", []int{0, 2, 1, 3}, "q_bhtd")
	prog.Reshape("flat", []int{ropeIndexedTestB, ropeIndexedTestT, ropeIndexedTestH, ropeIndexedTestHeadDim}, "k_bthd")
	prog.Transpose("k_bthd", []int{0, 2, 1, 3}, "k_bhtd")

	if indexed {
		prog.RoPEIndexed("q_bhtd", "k_bhtd", "positions", "qr", "kr", ropeIndexedTestT, ropeIndexedTestHeadDim, 0, ropeIndexedTestBase)
	} else {
		prog.RoPE("q_bhtd", "k_bhtd", "qr", "kr", ropeIndexedTestT, ropeIndexedTestHeadDim, 0, ropeIndexedTestBase)
	}

	prog.Transpose("qr", []int{0, 2, 1, 3}, "qr_bthd")
	prog.Reshape("qr_bthd", []int{ropeIndexedTestB * ropeIndexedTestT, ropeIndexedTestD}, "qr_flat")
	prog.MatMul("qr_flat", "w1", "logits")
	prog.CrossEntropy("logits", "targets", "loss")
	return prog
}

func runRoPEIndexedTestProgram(t *testing.T, prog *ir.Program, positions []int32) []float32 {
	t.Helper()

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0Data := make([]float32, ropeIndexedTestVocab*ropeIndexedTestD)
	for i := range w0Data {
		w0Data[i] = 0.1 + float32(i)*0.01
	}
	w0, err := FromData(w0Data, ropeIndexedTestVocab, ropeIndexedTestD)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	w1Data := make([]float32, ropeIndexedTestD*ropeIndexedTestVocab)
	for i := range w1Data {
		w1Data[i] = -0.05 + float32(i)*0.005
	}
	w1, err := FromData(w1Data, ropeIndexedTestD, ropeIndexedTestVocab)
	if err != nil {
		t.Fatalf("FromData w1: %v", err)
	}
	defer FreeHandle(w1)

	trainer, err := CreateTrainer(gpuProg, []int64{w0, w1}, ropeIndexedTestOptimizerSpec(2))
	if err != nil {
		t.Fatalf("CreateTrainer: %v", err)
	}
	defer TrainerDestroy(trainer)

	inputs := []TensorInput{
		{
			Name:  "tokens",
			DType: TensorInt32,
			Shape: []int{ropeIndexedTestB, ropeIndexedTestT},
			Data:  []int32{0, 1, 2, 3},
		},
		{
			Name:  "targets",
			DType: TensorInt32,
			Shape: []int{ropeIndexedTestB * ropeIndexedTestT},
			Data:  []int32{1, 2, 3, 0},
		},
	}
	if positions != nil {
		inputs = append(inputs, TensorInput{
			Name:  "positions",
			DType: TensorInt32,
			Shape: []int{ropeIndexedTestT},
			Data:  positions,
		})
	}

	if _, err := TrainerEvaluate(trainer, inputs); err != nil {
		t.Fatalf("TrainerEvaluate: %v", err)
	}

	qr, err := TrainerReadOutput(trainer, "qr", []int{
		ropeIndexedTestB,
		ropeIndexedTestH,
		ropeIndexedTestT,
		ropeIndexedTestHeadDim,
	})
	if err != nil {
		t.Fatalf("TrainerReadOutput(qr): %v", err)
	}
	return qr
}

func TestRoPEIndexedContiguousMatchesStandard(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	standardQR := runRoPEIndexedTestProgram(t, buildRoPEIndexedTestProgram(false), nil)
	indexedQR := runRoPEIndexedTestProgram(t, buildRoPEIndexedTestProgram(true), []int32{0, 1, 2, 3})

	requireSameFloat32s(t, "standard RoPE vs RoPEIndexed contiguous", standardQR, indexedQR, ropeIndexedTestTol)
}

func TestRoPEIndexedSparsePositionsUseAbsolutePositionValues(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	contiguousQR := runRoPEIndexedTestProgram(t, buildRoPEIndexedTestProgram(true), []int32{0, 1, 2, 3})
	sparseQR := runRoPEIndexedTestProgram(t, buildRoPEIndexedTestProgram(true), []int32{0, 4, 8, 12})

	for h := 0; h < ropeIndexedTestH; h++ {
		for d := 0; d < ropeIndexedTestHeadDim; d++ {
			i := ropeIndexedTestQRIndex(0, h, 0, d)
			if diff := math.Abs(float64(contiguousQR[i] - sparseQR[i])); diff > ropeIndexedTestTol {
				t.Fatalf("position 0 mismatch at head=%d dim=%d: contiguous=%g sparse=%g diff=%g",
					h, d, contiguousQR[i], sparseQR[i], diff)
			}
		}
	}

	var maxSparseDiff float64
	for h := 0; h < ropeIndexedTestH; h++ {
		for pos := 1; pos < ropeIndexedTestT; pos++ {
			for d := 0; d < ropeIndexedTestHeadDim; d++ {
				i := ropeIndexedTestQRIndex(0, h, pos, d)
				diff := math.Abs(float64(contiguousQR[i] - sparseQR[i]))
				if diff > maxSparseDiff {
					maxSparseDiff = diff
				}
			}
		}
	}
	if maxSparseDiff <= ropeIndexedTestTol {
		t.Fatalf("sparse positions produced the same qr rotations as contiguous positions; max diff=%g", maxSparseDiff)
	}
}

func TestRoPEPartialLeavesPassThroughDimsUnchanged(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := buildRoPEIndexedTestProgram(false)
	for i := range prog.Ops {
		if prog.Ops[i].Code == ir.OpRoPE {
			prog.Ops[i].IntParams[2] = 2
			break
		}
	}
	qr := runRoPEIndexedTestProgram(t, prog, nil)

	for tok := 0; tok < ropeIndexedTestT; tok++ {
		for h := 0; h < ropeIndexedTestH; h++ {
			for d := 2; d < ropeIndexedTestHeadDim; d++ {
				i := ropeIndexedTestQRIndex(0, h, tok, d)
				flatDim := h*ropeIndexedTestHeadDim + d
				want := 0.1 + float32(tok*ropeIndexedTestD+flatDim)*0.01
				if diff := math.Abs(float64(qr[i] - want)); diff > ropeIndexedTestTol {
					t.Fatalf("partial RoPE changed pass-through dim token=%d head=%d dim=%d: got=%g want=%g diff=%g",
						tok, h, d, qr[i], want, diff)
				}
			}
		}
	}
}

func requireSameFloat32s(t *testing.T, label string, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length mismatch: want=%d got=%d", label, len(want), len(got))
	}
	for i := range want {
		if diff := math.Abs(float64(want[i] - got[i])); diff > tol {
			t.Fatalf("%s mismatch at elem %d: want=%g got=%g diff=%g", label, i, want[i], got[i], diff)
		}
	}
}

func ropeIndexedTestQRIndex(b, h, tPos, d int) int {
	return (((b*ropeIndexedTestH+h)*ropeIndexedTestT+tPos)*ropeIndexedTestHeadDim + d)
}

func ropeIndexedTestOptimizerSpec(nWeights int) TrainerOptimizerSpec {
	weights := make([]WeightOptimizer, nWeights)
	for i := range weights {
		weights[i] = WeightOptimizer{GroupIndex: 0}
	}
	return TrainerOptimizerSpec{
		Groups: []OptimizerGroup{{
			Kind:    OptimizerAdamW,
			LR:      1e-3,
			Beta1:   0.9,
			Beta2:   0.95,
			Epsilon: 1e-8,
		}},
		Weights:       weights,
		DefaultBaseLR: 1e-3,
	}
}
