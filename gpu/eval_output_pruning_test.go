//go:build mlx && cgo

package gpu

import (
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestEvalProgramOutputPrunesUnrequestedBranches(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("wanted", ir.TensorFloat32, []int{2})
	prog.ScalarMul("x", 2, "wanted")
	prog.MatMul("missing_input", "w0", "unused")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{1}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)

	got, err := EvalProgramOutput(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{2}, Data: []float32{3, -4}},
	}, "wanted")
	if err != nil {
		t.Fatalf("EvalProgramOutput: %v", err)
	}
	if len(got) != 2 || got[0] != 6 || got[1] != -8 {
		t.Fatalf("wanted=%v, want [6 -8]", got)
	}
}

func TestEvalProgramOutputsPrunesUnrequestedBranches(t *testing.T) {
	if !Available() {
		t.Skip("MLX backend not available")
	}
	prog := ir.NewProgram(1)
	prog.DeclareInput("x", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("a", ir.TensorFloat32, []int{2})
	prog.DeclareOutput("b", ir.TensorFloat32, []int{2})
	prog.ScalarMul("x", 2, "a")
	prog.ScalarMul("x", -1, "b")
	prog.MatMul("missing_input", "w0", "unused")

	gpuProg, err := LowerIRProgram(prog)
	if err != nil {
		t.Fatalf("LowerIRProgram: %v", err)
	}
	defer gpuProg.Destroy()
	dummy, err := FromData([]float32{1}, 1, 1)
	if err != nil {
		t.Fatalf("FromData: %v", err)
	}
	defer FreeHandle(dummy)

	outs, err := EvalProgramOutputs(gpuProg, []int64{dummy}, []TensorInput{
		{Name: "x", DType: TensorFloat32, Shape: []int{2}, Data: []float32{3, -4}},
	}, []string{"a", "b"}, []int{2, 2})
	if err != nil {
		t.Fatalf("EvalProgramOutputs: %v", err)
	}
	if got := outs["a"]; len(got) != 2 || got[0] != 6 || got[1] != -8 {
		t.Fatalf("a=%v, want [6 -8]", got)
	}
	if got := outs["b"]; len(got) != 2 || got[0] != -3 || got[1] != 4 {
		t.Fatalf("b=%v, want [-3 4]", got)
	}
}
