package arch

import "testing"

func testFirstByteMaskConfig(enabled bool) *ArchConfig {
	return &ArchConfig{
		Name:      "first_byte_mask_test",
		ModelDim:  16,
		VocabSize: 256,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{
			Steps:         10,
			LR:            1e-3,
			BatchTokens:   8,
			FirstByteMask: enabled,
		},
	}
}

func TestParseArchConfig_FirstByteMask(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "first-byte-mask",
		"model_dim": 16,
		"vocab_size": 256,
		"seq_len": 4,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"first_byte_mask": true}
	}`), "test.json")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !cfg.Training.FirstByteMask {
		t.Fatal("training.first_byte_mask = false, want true")
	}
}

func TestBuildTrainingIRProgram_FirstByteMaskUsesMaskedLoss(t *testing.T) {
	prog, err := BuildIRProgramFromConfig(testFirstByteMaskConfig(true))
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpFirstByteMaskedCE); n != 1 {
		t.Fatalf("masked CE op count = %d, want 1", n)
	}
	if n := countOps(prog, OpCrossEntropy); n != 1 {
		t.Fatalf("unmasked CE op count = %d, want eval_loss op only", n)
	}
	if !declaresInput(prog, "first_byte_valid") {
		t.Fatal("missing first_byte_valid input")
	}
	if !declaresOutput(prog, "eval_loss") {
		t.Fatal("missing unmasked eval_loss output")
	}
}

func TestBuildEvalIRProgram_FirstByteMaskUsesUnmaskedLoss(t *testing.T) {
	prog, err := BuildEvalIRProgramFromConfig(testFirstByteMaskConfig(true))
	if err != nil {
		t.Fatalf("BuildEvalIRProgramFromConfig: %v", err)
	}
	if n := countOps(prog, OpFirstByteMaskedCE); n != 0 {
		t.Fatalf("eval masked CE op count = %d, want 0", n)
	}
	if declaresInput(prog, "first_byte_valid") {
		t.Fatal("eval program should not require first_byte_valid input")
	}
	if declaresOutput(prog, "eval_loss") {
		t.Fatal("eval program should not emit training-only eval_loss")
	}
}

func declaresInput(prog *Program, name string) bool {
	for _, in := range prog.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}

func declaresOutput(prog *Program, name string) bool {
	for _, out := range prog.Outputs {
		if out.Name == name {
			return true
		}
	}
	return false
}
