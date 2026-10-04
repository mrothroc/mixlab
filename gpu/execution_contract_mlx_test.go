//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"math"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestExecutionContractScanZeroResetOracle(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	p := ir.NewProgram(1)
	p.DeclareInput("x", ir.TensorFloat32, []int{3, 1})
	p.Scan("x", "w0", "carry", 1, 3, 1)
	p.DeclareOutput("carry", ir.TensorFloat32, []int{3, 1})
	g, err := LowerIRProgram(p)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()
	w, err := FromData([]float32{0}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandle(w)
	inputs := []TensorInput{{Name: "x", DType: TensorFloat32, Shape: []int{3, 1}, Data: []float32{1, 2, 4}}}
	for call := 0; call < 2; call++ {
		got, err := EvalProgramOutput(g, []int64{w}, inputs, "carry")
		if err != nil {
			t.Fatal(err)
		}
		maxErr := 0.0
		for i, want := range []float32{0.5, 1.25, 2.625} {
			diff := math.Abs(float64(got[i] - want))
			maxErr = math.Max(maxErr, diff)
			if diff > 1e-6+1e-6*math.Abs(float64(want)) {
				t.Fatalf("call %d carry[%d]=%g want %g", call, i, got[i], want)
			}
		}
		t.Logf("call=%d max_absolute_error=%g", call, maxErr)
	}
}
