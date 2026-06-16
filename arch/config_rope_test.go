package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseArchConfig_RopeDims(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_rope_dims",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, RopeDims: 16}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_rope_dims")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].RopeDims != 16 {
		t.Fatalf("rope_dims = %d, want 16", got.Blocks[0].RopeDims)
	}
}

func TestParseArchConfig_RopeConvention(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_rope_convention",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, RopeDims: 16, RopeConvention: "half_rotation"}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_rope_convention")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].RopeConvention != "half_rotation" {
		t.Fatalf("rope_convention = %q, want half_rotation", got.Blocks[0].RopeConvention)
	}
}

func TestParseArchConfig_InvalidRopeConvention(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_invalid_rope_convention",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4, RopeConvention: "interleaved"}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test_invalid_rope_convention")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "rope_convention") {
		t.Fatalf("error = %q, want rope_convention", err.Error())
	}
}

func TestParseArchConfig_InvalidRopeDims(t *testing.T) {
	tests := []struct {
		name     string
		ropeDims int
		wantErr  string
	}{
		{name: "negative", ropeDims: -2, wantErr: "must be > 0"},
		{name: "odd", ropeDims: 3, wantErr: "must be even"},
		{name: "too large", ropeDims: 64, wantErr: "must be <= head_dim=32"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ArchConfig{
				Name:      "test_invalid_rope_dims",
				ModelDim:  128,
				VocabSize: 1024,
				SeqLen:    128,
				Blocks:    []BlockSpec{{Type: "plain", Heads: 4, RopeDims: tt.ropeDims}},
				Training:  TrainingSpec{Steps: 100, LR: 3e-4},
			}
			data, _ := json.Marshal(cfg)
			_, err := ParseArchConfig(data, "test_invalid_rope_dims")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}
}
