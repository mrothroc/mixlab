package arch

import (
	"strings"
	"testing"
)

func TestMLMHeadValidation(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "invalid",
			body:    `"mlm_head": "weird", "training": {"steps": 1, "batch_tokens": 8}`,
			wantErr: "mlm_head",
		},
		{
			name:    "requires masked path",
			body:    `"tie_embeddings": true, "mlm_head": "bert", "training": {"steps": 1, "batch_tokens": 8}`,
			wantErr: "no masked objective path",
		},
		{
			name:    "requires tied embeddings",
			body:    `"mlm_head": "bert", "training": {"steps": 1, "batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 1}`,
			wantErr: "tie_embeddings",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestBERTMLMHeadWeightLayoutAndIR(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"tie_embeddings": true,
		"mlm_head": "bert",
		"training": {"steps": 1, "batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 1}`)

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if n := len(metas); n < 3 {
		t.Fatalf("weight count=%d", n)
	}
	start := len(metas) - 3
	if metas[start].Name != MLMHeadDenseWeightName || metas[start+1].Name != MLMHeadDenseBiasName || metas[start+2].Name != MLMHeadOutputBiasName {
		t.Fatalf("tail metas = %#v", metas[start:])
	}
	if got, want := metas[start].Shape, []int{cfg.ModelDim, cfg.ModelDim}; !mlmHeadIntSlicesEqual(got, want) {
		t.Fatalf("dense shape=%v want %v", got, want)
	}
	if got, want := metas[start+2].Shape, []int{cfg.VocabSize}; !mlmHeadIntSlicesEqual(got, want) {
		t.Fatalf("output bias shape=%v want %v", got, want)
	}

	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{Objective: ObjectiveMLM})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(metas) {
		t.Fatalf("program weights=%d metas=%d", prog.NumWeights, len(metas))
	}
	if got := weightInputForOutput(t, prog, OpMatMul, "mlm_head_dense_mm"); got != weightName(start) {
		t.Fatalf("mlm_head_dense weight=%q want %q", got, weightName(start))
	}
	if got := weightInputForOutput(t, prog, OpAdd, "mlm_head_dense_out"); got != weightName(start+1) {
		t.Fatalf("mlm_head_dense_bias weight=%q want %q", got, weightName(start+1))
	}
	if got := weightInputForOutput(t, prog, OpAdd, "mlm_head_logits"); got != weightName(start+2) {
		t.Fatalf("mlm_head_output_bias weight=%q want %q", got, weightName(start+2))
	}
	if !hasOpOutput(prog, OpTranspose, "mlm_head_tied_weight") {
		t.Fatalf("BERT MLM head did not transpose embedding weight for tied output")
	}
	if !hasOpOutput(prog, OpScalarMul, "logits") {
		t.Fatalf("BERT MLM head did not publish logits output")
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if evalProg.NumWeights != len(metas) {
		t.Fatalf("eval program weights=%d metas=%d", evalProg.NumWeights, len(metas))
	}
	if hasOpOutput(evalProg, OpMatMul, "mlm_head_dense_mm") {
		t.Fatalf("eval program should reserve but not execute BERT MLM transform")
	}
	if got := weightInputForOutput(t, evalProg, OpMatMul, "logits"); got != "tied_head" {
		t.Fatalf("eval logits matmul input=%q want tied_head", got)
	}
}

func mlmHeadIntSlicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func hasOpOutput(prog *Program, code int, output string) bool {
	for _, op := range prog.Ops {
		if op.Code != code {
			continue
		}
		for _, out := range op.Outputs {
			if out == output {
				return true
			}
		}
	}
	return false
}
