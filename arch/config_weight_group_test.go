package arch

import (
	"strings"
	"testing"
)

func TestWeightGroupParsing(t *testing.T) {
	cfgJSON := `{
		"name": "wg",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [
			{"type": "plain", "heads": 4, "weight_group": "shared_attn"},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4, "weight_group": "shared_attn"}
		],
		"training": {"batch_tokens": 32}
	}`

	cfg, err := ParseArchConfig([]byte(cfgJSON), "weight_group_parse")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got := cfg.Blocks[0].WeightGroup; got != "shared_attn" {
		t.Fatalf("blocks[0].WeightGroup=%q want shared_attn", got)
	}
	if got := cfg.Blocks[2].WeightGroup; got != "shared_attn" {
		t.Fatalf("blocks[2].WeightGroup=%q want shared_attn", got)
	}
}

func TestWeightGroupMismatchedType(t *testing.T) {
	cfgJSON := `{
		"name": "wg",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [
			{"type": "plain", "heads": 4, "weight_group": "shared"},
			{"type": "swiglu", "weight_group": "shared"}
		],
		"training": {"batch_tokens": 32}
	}`

	_, err := ParseArchConfig([]byte(cfgJSON), "weight_group_mismatch")
	if err == nil {
		t.Fatal("expected error for mismatched weight_group types")
	}
	if !strings.Contains(err.Error(), "weight_group") || !strings.Contains(err.Error(), "type mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestWeightGroupEmpty(t *testing.T) {
	cfgJSON := `{
		"name": "wg",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu", "weight_group": ""},
			{"type": "plain", "heads": 4}
		],
		"training": {"batch_tokens": 32}
	}`

	cfg, err := ParseArchConfig([]byte(cfgJSON), "weight_group_empty")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if cfg.Blocks[1].WeightGroup != "" {
		t.Fatalf("blocks[1].WeightGroup=%q want empty", cfg.Blocks[1].WeightGroup)
	}

	base := &ArchConfig{
		Name:      "base",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
		},
		Training: TrainingSpec{BatchTokens: 32},
	}
	progBase, err := BuildIRProgramFromConfig(base)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(base): %v", err)
	}
	progEmpty, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(empty): %v", err)
	}
	if progEmpty.NumWeights != progBase.NumWeights {
		t.Fatalf("NumWeights=%d want %d", progEmpty.NumWeights, progBase.NumWeights)
	}
}
