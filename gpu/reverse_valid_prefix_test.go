//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestReverseValidPrefixMatchesCPUOracle(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX backend not available")
	}
	const B, T, D = 2, 5, 2
	input := []float32{
		1, 2, 3, 4, 5, 6, 70, 80, 90, 100,
		11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	}
	mask := []float32{
		1, 1, 1, 0, 0,
		1, 1, 1, 1, 1,
	}
	want := []float32{
		5, 6, 3, 4, 1, 2, 0, 0, 0, 0,
		19, 20, 17, 18, 15, 16, 13, 14, 11, 12,
	}

	prog := ir.NewProgram(1)
	prog.DeclareInput("input", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareInput("valid", ir.TensorFloat32, []int{B, T})
	prog.ReverseValidPrefix("input", "valid", "reversed", B, T)
	prog.ReverseValidPrefix("reversed", "valid", "round_trip", B, T)
	prog.DeclareOutput("reversed", ir.TensorFloat32, []int{B * T, D})
	prog.DeclareOutput("round_trip", ir.TensorFloat32, []int{B * T, D})
	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatal(err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromDataShape([]float32{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	defer FreeHandle(dummy)
	inputs := []TensorInput{
		{Name: "input", DType: TensorFloat32, Shape: []int{B * T, D}, Data: input},
		{Name: "valid", DType: TensorFloat32, Shape: []int{B, T}, Data: mask},
	}
	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "reversed")
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffFloat32(got, want); diff != 0 {
		t.Fatalf("reverse L_inf=%g\ngot=%v\nwant=%v", diff, got, want)
	}
	roundTrip, err := EvalProgramOutput(gpuProg, []int64{dummy}, inputs, "round_trip")
	if err != nil {
		t.Fatal(err)
	}
	wantRoundTrip := append([]float32(nil), input...)
	for i := 3 * D; i < 5*D; i++ {
		wantRoundTrip[i] = 0
	}
	if diff := maxAbsDiffFloat32(roundTrip, wantRoundTrip); diff != 0 {
		t.Fatalf("round trip L_inf=%g\ngot=%v\nwant=%v", diff, roundTrip, wantRoundTrip)
	}
}
