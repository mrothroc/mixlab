package arch

import "testing"

func TestPlainAttentionDropoutAfterSoftmaxBeforeValueMatMul(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "attn_dropout_ir",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"dropout": 0.1,
		"hidden_dropout": 0,
		"attn_dropout": 0.25,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16}
	}`), "attn_dropout_ir")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	softmaxIdx := -1
	dropoutIdx := -1
	ctxMatMulIdx := -1
	for i, op := range prog.Ops {
		switch op.Code {
		case OpSoftmax:
			if len(op.Outputs) == 1 && op.Outputs[0] == "x_attn_0_attn" {
				softmaxIdx = i
			}
		case OpDropout:
			if len(op.Inputs) == 1 && op.Inputs[0] == "x_attn_0_attn" {
				dropoutIdx = i
				if len(op.FloatParams) != 1 || op.FloatParams[0] != 0.25 {
					t.Fatalf("attention dropout params=%v, want [0.25]", op.FloatParams)
				}
				if len(op.Outputs) != 1 || op.Outputs[0] != "x_attn_0_attn_dropout" {
					t.Fatalf("attention dropout output=%v", op.Outputs)
				}
			}
		case OpMatMul:
			if len(op.Inputs) == 2 && op.Inputs[0] == "x_attn_0_attn_dropout" && op.Inputs[1] == "x_attn_0_vh" {
				ctxMatMulIdx = i
			}
		}
	}
	if softmaxIdx == -1 || dropoutIdx == -1 || ctxMatMulIdx == -1 {
		t.Fatalf("missing softmax/dropout/ctx matmul indices: softmax=%d dropout=%d ctx=%d", softmaxIdx, dropoutIdx, ctxMatMulIdx)
	}
	if softmaxIdx >= dropoutIdx || dropoutIdx >= ctxMatMulIdx {
		t.Fatalf("unexpected order softmax=%d dropout=%d ctx=%d", softmaxIdx, dropoutIdx, ctxMatMulIdx)
	}
	if got := countOps(prog, OpDropout); got != 1 {
		t.Fatalf("training dropout op count=%d, want only attention dropout", got)
	}

	evalProg, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if got := countOps(evalProg, OpDropout); got != 0 {
		t.Fatalf("eval dropout op count=%d, want 0", got)
	}
}
