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

func TestTTTDiagnosticsAreSeparatedAndFormattedDeterministically(t *testing.T) {
	values := map[string]float64{
		"primary_loss":                  2.5,
		"block_0_ttt_state_drift":       0.125,
		"block_0_ttt_inner_loss_before": 0.75,
	}
	losses, extra := splitTrainingDiagnostics(values)
	if !reflect.DeepEqual(losses, map[string]float64{"primary_loss": 2.5}) {
		t.Fatalf("losses=%v", losses)
	}
	want := "block_0_ttt_inner_loss_before=0.75 block_0_ttt_state_drift=0.125"
	if got := formatTrainingExtraDiagnostics(extra); got != want {
		t.Fatalf("formatted=%q want %q", got, want)
	}
}

func TestTTTDiagnosticsAreExcludedFromTrainingStepOutputs(t *testing.T) {
	prog := arch.NewProgram(1)
	prog.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("invariance_loss", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("block_0_ttt_inner_loss_before", arch.TensorFloat32, []int{1})
	prog.DeclareOutput("block_0_ttt_state_drift", arch.TensorFloat32, []int{1})

	if got, want := declaredTrainingStepComponentLossOutputs(prog), []string{"invariance_loss"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("training-step outputs=%v, want %v", got, want)
	}
	if got, want := declaredTTTDiagnosticOutputs(prog), []string{"block_0_ttt_inner_loss_before", "block_0_ttt_state_drift"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("TTT diagnostic outputs=%v, want %v", got, want)
	}
}
