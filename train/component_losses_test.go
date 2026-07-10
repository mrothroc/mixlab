package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestDeclaredComponentLossOutputsUsesPublicScalarLossDeclarations(t *testing.T) {
	prog := arch.NewProgram(1)
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("eval_loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("invariance_loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("head_scorer_loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("per_token_nll", arch.TensorFloat32, []int{8})
	prog.DeclareOutput("not_a_loss", arch.TensorFloat32, []int{2})

	got := declaredComponentLossOutputs(prog)
	want := []string{"invariance_loss", "head_scorer_loss"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("component outputs=%v, want %v", got, want)
	}
}

func TestDeclaredComponentLossOutputsNoOpProgramIsEmpty(t *testing.T) {
	prog := arch.NewProgram(1)
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	if got := declaredComponentLossOutputs(prog); len(got) != 0 {
		t.Fatalf("component outputs=%v, want none", got)
	}
}
