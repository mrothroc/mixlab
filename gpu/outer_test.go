//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func buildOuterVectorTestProgram() *ir.Program {
	prog := ir.NewProgram(1)
	prog.DeclareInput("a", ir.TensorFloat32, []int{3})
	prog.DeclareInput("b", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{3, 2})
	prog.Outer("a", "b", "out")
	return prog
}

func TestOuterSupportsRank1Inputs(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	gpuProg, err := LowerIRProgram(buildOuterVectorTestProgram())
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	inputs := []TensorInput{
		{
			Name:  "a",
			DType: TensorFloat32,
			Shape: []int{3},
			Data:  []float32{0, 1, 2},
		},
		{
			Name:  "b",
			DType: TensorFloat32,
			Shape: []int{2},
			Data:  []float32{10, 20},
		},
	}

	out, err := evalProgramOutput(gpuProg, []int64{w0}, inputs, "out")
	if err != nil {
		t.Fatalf("evalProgramOutput(out): %v", err)
	}

	want := []float32{
		0, 0,
		10, 20,
		20, 40,
	}
	if len(out) != len(want) {
		t.Fatalf("output len=%d, want %d", len(out), len(want))
	}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-5 {
			t.Fatalf("out[%d]=%g, want %g", i, out[i], want[i])
		}
	}
}
