//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestDivSafeHonorsEpsilon(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("num", ir.TensorFloat32, []int{2})
	prog.DeclareInput("den", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{2})
	prog.DivSafe("num", "den", 0.5, "out")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	w0, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatalf("FromData w0: %v", err)
	}
	defer FreeHandle(w0)

	out, err := evalProgramOutput(gpuProg, []int64{w0}, []TensorInput{
		{Name: "num", DType: TensorFloat32, Shape: []int{2}, Data: []float32{1, -2}},
		{Name: "den", DType: TensorFloat32, Shape: []int{2}, Data: []float32{0, 1}},
	}, "out")
	if err != nil {
		t.Fatalf("evalProgramOutput(out): %v", err)
	}

	want := []float32{2, -4.0 / 3.0}
	for i := range want {
		if diff := math.Abs(float64(out[i] - want[i])); diff > 1e-5 {
			t.Fatalf("out[%d]=%g, want %g", i, out[i], want[i])
		}
	}
}
