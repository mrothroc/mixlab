//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestGridDilationInvalidRuntimeParams(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	for _, tc := range []struct {
		name   string
		op     int
		params []int
	}{
		{"zero", ir.OpConv2D, []int{3, 1, 1, 0}},
		{"negative", ir.OpConv2D, []int{3, 1, 1, -1}},
		{"overflow", ir.OpConv2D, []int{3, 1, 1, math.MaxInt32}},
		{"transpose", ir.OpConvTranspose2D, []int{3, 1, 1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := ir.NewProgram(2)
			p.AddOp(tc.op, []string{"w0", "w1"}, []string{"y"}, nil, tc.params)
			p.DeclareOutput("y", ir.TensorFloat32, []int{1, 3, 3, 1})
			gp, err := LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Destroy()
			x, err := FromDataShape(make([]float32, 9), []int{1, 3, 3, 1})
			if err != nil {
				t.Fatal(err)
			}
			defer FreeHandles([]int64{x})
			w, err := FromDataShape(make([]float32, 9), []int{1, 3, 3, 1})
			if err != nil {
				t.Fatal(err)
			}
			defer FreeHandles([]int64{w})
			if _, err = EvalProgramOutput(gp, []int64{x, w}, nil, "y"); err == nil {
				t.Fatal("invalid raw IR was accepted")
			}
		})
	}
}
