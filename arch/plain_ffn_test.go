package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEmitPlainAttentionIR_GatedFFNTail(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, FFNActivation: "geglu"}, "x", 0, 128, 64, 2, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}
	if got := countOps(p, OpGELU); got != 1 {
		t.Fatalf("GELU ops=%d want 1", got)
	}
	if got := countOps(p, OpSiLU); got != 0 {
		t.Fatalf("SiLU ops=%d want 0 for GEGLU tail", got)
	}
	if got := weightInputForOutput(t, p, OpMatMul, "x_attn_0_ff_gate"); got != "w5" {
		t.Fatalf("ff_gate weight=%q want w5", got)
	}
	if got := weightInputForOutput(t, p, OpMatMul, "x_attn_0_ff1"); got != "w6" {
		t.Fatalf("ff1 weight=%q want w6", got)
	}
	if got := weightInputForOutput(t, p, OpMatMul, "x_attn_0_ff2"); got != "w7" {
		t.Fatalf("ff2 weight=%q want w7", got)
	}
}

func TestEmitPlainAttentionIR_SwiGLUTail(t *testing.T) {
	p := NewProgram(8)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, FFNActivation: "swiglu"}, "x", 0, 128, 64, 2, 1024, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 8 {
		t.Fatalf("expected wi=8, got %d", wi)
	}
	if got := countOps(p, OpGELU); got != 0 {
		t.Fatalf("GELU ops=%d want 0 for SwiGLU tail", got)
	}
	if got := countOps(p, OpSiLU); got != 1 {
		t.Fatalf("SiLU ops=%d want 1 for SwiGLU tail", got)
	}
	if got := weightInputForOutput(t, p, OpMatMul, "x_attn_0_ff_gate"); got != "w5" {
		t.Fatalf("ff_gate weight=%q want w5", got)
	}
}

func TestPlainFFNActivationConfigParsing(t *testing.T) {
	cfg := `{
		"name": "test",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [{"type": "plain", "heads": 4, "ffn_activation": "geglu"}]
	}`
	got, err := ParseArchConfig([]byte(cfg), "test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].FFNActivation != "geglu" {
		t.Fatalf("ffn_activation = %q, want geglu", got.Blocks[0].FFNActivation)
	}
}

func TestPlainFFNActivationValidation(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    32,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, FFNActivation: "relu"}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for invalid plain ffn_activation")
	}
	if !strings.Contains(err.Error(), "ffn_activation") {
		t.Errorf("error should mention ffn_activation: %v", err)
	}
}
