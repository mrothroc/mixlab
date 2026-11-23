//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestGridRectangularPadPyTorchForwardBackward(t *testing.T) {
	if !runInStrictFP32Process(t) {
		return
	}
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	raw, err := os.ReadFile("testdata/grid_rectangular_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct{ Cases []gridReferenceCase }
	if err = json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	runGridReferenceCases(t, fixtures.Cases)
}

func TestGridRectangularPadInvalidIR(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	for _, tc := range []struct {
		name   string
		code   int
		params []int
	}{
		{"short", ir.OpConv2D, []int{1, 1, 1, 1, 0}},
		{"negative", ir.OpConv2D, []int{1, 1, 1, 1, 0, -1, 1, 1}},
		{"zero_stride", ir.OpConv2D, []int{1, 1, 0, 1, 0, 0, 1, 1}},
		{"overflow", ir.OpConv2D, []int{1, 1, 1, 1, math.MaxInt32, 0, 1, 1}},
		{"transpose_pairs", ir.OpConvTranspose2D, []int{1, 1, 1, 1, 0, 0, 1, 1}},
		{"pad_short", ir.OpSpatialPad, []int{0, 0, 0, 0}},
		{"pad_mode", ir.OpSpatialPad, []int{0, 0, 0, 0, 3}},
		{"pad_negative", ir.OpSpatialPad, []int{-1, 0, 0, 0, 0}},
		{"pad_wrap", ir.OpSpatialPad, []int{3, 0, 0, 0, 2}},
		{"pad_overflow", ir.OpSpatialPad, []int{math.MaxInt32, 0, 0, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := ir.NewProgram(2)
			names := []string{"w0", "w1"}
			if tc.code == ir.OpSpatialPad {
				names = names[:1]
			}
			p.AddOp(tc.code, names, []string{"y"}, nil, tc.params)
			p.DeclareOutput("y", ir.TensorFloat32, []int{1, 2, 3, 1})
			gp, err := LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Destroy()
			x, err := FromDataShape(make([]float32, 6), []int{1, 2, 3, 1})
			if err != nil {
				t.Fatal(err)
			}
			w, err := FromDataShape([]float32{1}, []int{1, 1, 1, 1})
			if err != nil {
				FreeHandles([]int64{x})
				t.Fatal(err)
			}
			defer FreeHandles([]int64{x, w})
			if _, err = EvalProgramOutput(gp, []int64{x, w}, nil, "y"); err == nil {
				t.Fatal("accepted invalid spatial IR")
			}
		})
	}
}
