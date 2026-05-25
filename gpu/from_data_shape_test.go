//go:build mlx && cgo

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestFromDataShapePreservesRankedTensor(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}

	const (
		b = 2
		k = 3
		d = 4
	)
	data := make([]float32, b*k*d)
	for i := range data {
		data[i] = float32(i) / 10
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("dummy", ir.TensorFloat32, []int{1})
	prog.DeclareOutput("out", ir.TensorFloat32, []int{b, k, d})
	prog.StopGradient("w0", "out")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()

	weight, err := FromDataShape(data, []int{b, k, d})
	if err != nil {
		t.Fatalf("FromDataShape: %v", err)
	}
	defer FreeHandle(weight)

	got, err := EvalProgramOutput(gpuProg, []int64{weight}, []TensorInput{{
		Name:  "dummy",
		DType: TensorFloat32,
		Shape: []int{1},
		Data:  []float32{0},
	}}, "out")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	if len(got) != len(data) {
		t.Fatalf("output len=%d want %d", len(got), len(data))
	}
	for i := range data {
		if got[i] != data[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], data[i])
		}
	}
}
