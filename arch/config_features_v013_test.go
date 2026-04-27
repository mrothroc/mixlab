package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseArchConfig_WindowSize(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_window_size",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, WindowSize: 128}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_window_size")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].WindowSize != 128 {
		t.Fatalf("window_size = %d, want 128", got.Blocks[0].WindowSize)
	}
}

func TestParseArchConfig_SparseAttnGate(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_sparse_attn_gate",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, SparseAttnGate: true}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_sparse_attn_gate")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.Blocks[0].SparseAttnGate {
		t.Fatal("sparse_attn_gate = false, want true")
	}
}

func TestParseArchConfig_InvalidWindowSize(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_invalid_window_size",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, WindowSize: -1}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test_invalid_window_size")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "window_size") {
		t.Fatalf("error = %q, want substring %q", err.Error(), "window_size")
	}
}
