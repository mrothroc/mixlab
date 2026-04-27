//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

const (
	causalMaskTestT      = 8
	causalMaskTestWindow = 4
	causalMaskMaskValue  = -1e9
	causalMaskTol        = 1e-3
)

func buildCausalMaskTestProgram() *ir.Program {
	prog := ir.NewProgram(1)
	prog.DeclareInput("scores", ir.TensorFloat32, []int{1, 1, causalMaskTestT, causalMaskTestT})
	prog.DeclareInput("targets", ir.TensorInt32, []int{causalMaskTestT})
	prog.DeclareOutput("masked", ir.TensorFloat32, []int{1, 1, causalMaskTestT, causalMaskTestT})
	prog.DeclareOutput("loss", ir.TensorFloat32, []int{1})

	prog.CausalMask("scores", causalMaskTestT, causalMaskTestWindow, "masked")
	prog.Reshape("masked", []int{causalMaskTestT, causalMaskTestT}, "masked_flat")
	prog.MatMul("masked_flat", "w0", "logits")
	prog.CrossEntropy("logits", "targets", "loss")
	return prog
}

func TestCausalMaskSlidingWindowPattern(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(buildCausalMaskTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData(make([]float32, causalMaskTestT), causalMaskTestT, 1)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	inputs := []TensorInput{
		{
			Name:  "scores",
			DType: TensorFloat32,
			Shape: []int{1, 1, causalMaskTestT, causalMaskTestT},
			Data:  make([]float32, causalMaskTestT*causalMaskTestT),
		},
		{
			Name:  "targets",
			DType: TensorInt32,
			Shape: []int{causalMaskTestT},
			Data:  make([]int32, causalMaskTestT),
		},
	}
	masked, err := evalProgramOutput(gpuProg, []int64{w0}, inputs, "masked")
	if err != nil {
		t.Fatalf("evalProgramOutput(masked): %v", err)
	}

	for q := 0; q < causalMaskTestT; q++ {
		minAllowed := q - (causalMaskTestWindow - 1)
		if minAllowed < 0 {
			minAllowed = 0
		}
		for k := 0; k < causalMaskTestT; k++ {
			got := masked[q*causalMaskTestT+k]
			allowed := k >= minAllowed && k <= q
			if allowed {
				if diff := math.Abs(float64(got)); diff > causalMaskTol {
					t.Fatalf("mask[%d,%d]=%g, want 0", q, k, got)
				}
				continue
			}
			if math.Abs(float64(got-causalMaskMaskValue)) > causalMaskTol {
				t.Fatalf("mask[%d,%d]=%g, want %g", q, k, got, float32(causalMaskMaskValue))
			}
		}
	}
}
