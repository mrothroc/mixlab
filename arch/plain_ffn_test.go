package arch

import (
	"encoding/json"
	"reflect"
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

func TestPlainAttentionBiasAndValueGateWeightLayout(t *testing.T) {
	metas, err := CollectWeightShapes(64, 512, 16, 1.0, false, false, false, false, []BlockSpec{
		{Type: "plain", Heads: 4, AttnBias: true, AttnValueGate: true},
	})
	if err != nil {
		t.Fatalf("CollectWeightShapes: %v", err)
	}
	want := map[string][]int{
		"wq":      {64, 64},
		"wq_bias": {64},
		"wk":      {64, 64},
		"wk_bias": {64},
		"wv":      {64, 128},
		"wv_bias": {128},
		"wo":      {64, 64},
		"wo_bias": {64},
	}
	got := map[string][]int{}
	for _, meta := range metas {
		if _, ok := want[meta.Name]; ok {
			got[meta.Name] = meta.Shape
		}
	}
	for name, shape := range want {
		if !reflect.DeepEqual(got[name], shape) {
			t.Fatalf("%s shape=%v want %v (metas=%+v)", name, got[name], shape, metas)
		}
	}
}

func TestEmitPlainAttentionIR_AttnBiasAndValueGate(t *testing.T) {
	p := NewProgram(11)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, AttnBias: true, AttnValueGate: true}, "x", 0, 64, 8, 2, 512, 0, nil, 1.0, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 11 {
		t.Fatalf("wi=%d want 11", wi)
	}
	if got := weightInputForOutput(t, p, OpMatMul, "x_attn_0_v_raw_matmul"); got != "w5" {
		t.Fatalf("value projection weight=%q want w5", got)
	}
	if got := countOps(p, OpGELU); got != 1 {
		t.Fatalf("GELU ops=%d want 1 for attention value gate", got)
	}
	foundValueSlice := false
	foundGateSlice := false
	foundGateApply := false
	foundWOBias := false
	for _, op := range p.Ops {
		switch op.Code {
		case OpSlice:
			if len(op.Inputs) == 1 && op.Inputs[0] == "x_attn_0_v_raw" && reflect.DeepEqual(op.IntParams, []int{0, 64, 1, 1}) {
				foundValueSlice = true
			}
			if len(op.Inputs) == 1 && op.Inputs[0] == "x_attn_0_v_raw" && reflect.DeepEqual(op.IntParams, []int{64, 128, 1, 1}) {
				foundGateSlice = true
			}
		case OpMul:
			if len(op.Inputs) == 2 && op.Inputs[0] == "x_attn_0_flat" && op.Inputs[1] == "x_attn_0_value_gate_act" {
				foundGateApply = true
			}
		case OpAdd:
			if len(op.Inputs) == 2 && op.Inputs[0] == "x_attn_0_proj_matmul" && op.Inputs[1] == "w8" {
				foundWOBias = true
			}
		}
	}
	if !foundValueSlice || !foundGateSlice || !foundGateApply || !foundWOBias {
		t.Fatalf("missing value-gate/bias ops: value_slice=%v gate_slice=%v gate_apply=%v wo_bias=%v", foundValueSlice, foundGateSlice, foundGateApply, foundWOBias)
	}
}
