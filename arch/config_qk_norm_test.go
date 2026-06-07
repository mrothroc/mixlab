package arch

import (
	"encoding/json"
	"testing"
)

func TestParseArchConfig_QKNorm(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_qk_norm",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, QKNorm: true}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_qk_norm")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.Blocks[0].QKNorm {
		t.Fatal("qk_norm = false, want true")
	}
}
