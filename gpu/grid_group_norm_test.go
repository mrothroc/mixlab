//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestGridGroupNormPyTorchForwardBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	b, err := os.ReadFile("testdata/grid_group_norm_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct{ Cases []gridReferenceCase }
	if err = json.Unmarshal(b, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures.Cases) != 20 {
		t.Fatalf("expected 20 GroupNorm reference cases, got %d", len(fixtures.Cases))
	}
	runGridReferenceCases(t, fixtures.Cases)
}

func TestGridGroupNormInvalidRuntime(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	for _, tc := range []struct {
		name   string
		groups int
		eps    float32
		shape  []int
	}{
		{"groups_zero", 0, 1e-5, []int{4}}, {"groups_negative", -1, 1e-5, []int{4}},
		{"groups_indivisible", 3, 1e-5, []int{4}}, {"eps_zero", 2, 0, []int{4}},
		{"eps_nan", 2, float32(math.NaN()), []int{4}}, {"eps_inf", 2, float32(math.Inf(1)), []int{4}},
		{"scale_rank", 2, 1e-5, []int{2, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := ir.NewProgram(3)
			p.AddOp(ir.OpGroupNorm, []string{"w0", "w1", "w2"}, []string{"y"}, []float32{tc.eps}, []int{tc.groups})
			p.DeclareOutput("y", ir.TensorFloat32, []int{1, 2, 3, 4})
			gp, err := LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Destroy()
			h := make([]int64, 3)
			defer FreeHandles(h)
			for i, s := range [][]int{{1, 2, 3, 4}, tc.shape, {4}} {
				n := 1
				for _, d := range s {
					n *= d
				}
				h[i], err = FromDataShape(make([]float32, n), s)
				if err != nil {
					t.Fatal(err)
				}
			}
			if _, err = EvalProgramOutput(gp, h, nil, "y"); err == nil {
				t.Fatal("invalid raw GroupNorm IR accepted")
			}
		})
	}
}
