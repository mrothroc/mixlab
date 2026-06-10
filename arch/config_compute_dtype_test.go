package arch

import (
	"strings"
	"testing"
)

func TestTrainingComputeDTypeDefaultsAndParses(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "compute_dtype_default",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"batch_tokens": 8}
	}`), "compute_dtype_default")
	if err != nil {
		t.Fatalf("ParseArchConfig(default): %v", err)
	}
	if cfg.Training.ComputeDType != "" {
		t.Fatalf("default compute_dtype stored as %q, want omitted/empty", cfg.Training.ComputeDType)
	}
	if got := cfg.Training.EffectiveComputeDType(); got != "float32" {
		t.Fatalf("effective default compute_dtype=%q, want float32", got)
	}

	cfg, err = ParseArchConfig([]byte(`{
		"name": "compute_dtype_bf16",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"batch_tokens": 8, "compute_dtype": " BF16 "}
	}`), "compute_dtype_bf16")
	if err != nil {
		t.Fatalf("ParseArchConfig(bf16): %v", err)
	}
	if cfg.Training.ComputeDType != "bf16" {
		t.Fatalf("compute_dtype=%q, want normalized bf16", cfg.Training.ComputeDType)
	}
	if got := cfg.Training.EffectiveComputeDType(); got != "bf16" {
		t.Fatalf("effective compute_dtype=%q, want bf16", got)
	}
}

func TestTrainingComputeDTypeInvalid(t *testing.T) {
	_, err := ParseArchConfig([]byte(`{
		"name": "compute_dtype_bad",
		"model_dim": 32,
		"vocab_size": 128,
		"seq_len": 8,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"batch_tokens": 8, "compute_dtype": "fp16"}
	}`), "compute_dtype_bad")
	if err == nil {
		t.Fatal("expected invalid compute_dtype error")
	}
	if !strings.Contains(err.Error(), "training.compute_dtype") {
		t.Fatalf("error=%q, want mention of training.compute_dtype", err.Error())
	}
}
