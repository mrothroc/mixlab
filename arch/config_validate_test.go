package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNegativeWeightDecay(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{WeightDecay: -0.01},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative weight_decay")
	}
	if !strings.Contains(err.Error(), "weight_decay") {
		t.Errorf("error should mention weight_decay: %v", err)
	}
}

func TestNegativePerGroupWeightDecay(t *testing.T) {
	tests := []struct {
		name  string
		field string
	}{
		{name: "embed", field: "embed_weight_decay"},
		{name: "matrix", field: "matrix_weight_decay"},
		{name: "scalar", field: "scalar_weight_decay"},
		{name: "head", field: "head_weight_decay"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{
				"model_dim": 128,
				"vocab_size": 1024,
				"blocks": [{"type": "plain", "heads": 4}],
				"training": {"` + tt.field + `": -0.01}
			}`)
			_, err := ParseArchConfig(raw, "test")
			if err == nil {
				t.Fatal("expected error for negative per-group weight decay")
			}
			if !strings.Contains(err.Error(), "per-group weight decay") {
				t.Errorf("error should mention per-group weight decay: %v", err)
			}
		})
	}
}

func TestNegativeGradClip(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{GradClip: -1.0},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative grad_clip")
	}
	if !strings.Contains(err.Error(), "grad_clip") {
		t.Errorf("error should mention grad_clip: %v", err)
	}
}

func TestNegativeSWAStart(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{SWAStart: -1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative swa_start")
	}
	if !strings.Contains(err.Error(), "swa_start") {
		t.Errorf("error should mention swa_start: %v", err)
	}
}

func TestInvalidSWADecay(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{SWADecay: 1.0},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for invalid swa_decay")
	}
	if !strings.Contains(err.Error(), "swa_decay") {
		t.Errorf("error should mention swa_decay: %v", err)
	}
}

func TestExplicitZeroSWADecayIsValid(t *testing.T) {
	raw := []byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"swa_decay": 0}
	}`)
	cfg, err := ParseArchConfig(raw, "test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if cfg.Training.SWADecay != 0 {
		t.Fatalf("swa_decay = %g, want explicit 0", cfg.Training.SWADecay)
	}
}

func TestInvalidSWAInterval(t *testing.T) {
	raw := []byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"swa_interval": 0}
	}`)
	_, err := ParseArchConfig(raw, "test")
	if err == nil {
		t.Fatal("expected error for invalid swa_interval")
	}
	if !strings.Contains(err.Error(), "swa_interval") {
		t.Errorf("error should mention swa_interval: %v", err)
	}
}

func TestNegativeWarmdownSteps(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{WarmdownSteps: -1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative warmdown_steps")
	}
	if !strings.Contains(err.Error(), "warmdown_steps") {
		t.Errorf("error should mention warmdown_steps: %v", err)
	}
}

func TestInvalidWarmupScheduleFields(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "negative warmup steps",
			body:    `"training": {"warmup_steps": -1}`,
			wantErr: "warmup_steps",
		},
		{
			name:    "negative warmup ratio",
			body:    `"training": {"warmup_ratio": -0.1}`,
			wantErr: "warmup_ratio",
		},
		{
			name:    "too large warmup ratio",
			body:    `"training": {"warmup_ratio": 1.1}`,
			wantErr: "warmup_ratio",
		},
		{
			name:    "ambiguous warmup",
			body:    `"training": {"warmup_steps": 10, "warmup_ratio": 0.1}`,
			wantErr: "warmup_steps and training.warmup_ratio",
		},
		{
			name:    "negative hold",
			body:    `"training": {"hold_steps": -1}`,
			wantErr: "hold_steps",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{
				"model_dim": 128,
				"vocab_size": 1024,
				"blocks": [{"type": "plain", "heads": 4}],
				` + tt.body + `
			}`)
			_, err := ParseArchConfig(raw, "test")
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error %q should mention %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestNegativeTargetValLoss(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{TargetValLoss: -0.1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative target_val_loss")
	}
	if !strings.Contains(err.Error(), "target_val_loss") {
		t.Errorf("error should mention target_val_loss: %v", err)
	}
}

func TestInvalidSplitDropoutFields(t *testing.T) {
	tests := []struct {
		name    string
		field   string
		wantErr string
	}{
		{name: "attn", field: `"attn_dropout": 1.1,`, wantErr: "attn_dropout"},
		{name: "hidden", field: `"hidden_dropout": -0.1,`, wantErr: "hidden_dropout"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := []byte(`{
				"name": "bad_dropout",
				"model_dim": 128,
				"vocab_size": 1024,
				"seq_len": 128,
				` + tt.field + `
				"blocks": [{"type": "plain", "heads": 4}],
				"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024}
			}`)
			_, err := ParseArchConfig(raw, "bad_dropout")
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestInvalidWeightInit(t *testing.T) {
	raw := []byte(`{
		"name": "bad_weight_init",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024, "weight_init": "weird"}
	}`)
	_, err := ParseArchConfig(raw, "bad_weight_init")
	if err == nil || !strings.Contains(err.Error(), "weight_init") {
		t.Fatalf("ParseArchConfig error=%v, want weight_init validation error", err)
	}
}

func TestGPT2WeightInitAccepted(t *testing.T) {
	raw := []byte(`{
		"name": "gpt2_weight_init",
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 100, "lr": 0.0003, "batch_tokens": 1024, "weight_init": "gpt2"}
	}`)
	cfg, err := ParseArchConfig(raw, "gpt2_weight_init")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if cfg.Training.WeightInit != "gpt2" {
		t.Fatalf("weight_init=%q, want gpt2", cfg.Training.WeightInit)
	}
}

func TestNegativeHardwareTFLOPs(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{HardwareTFLOPs: -1},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for negative hardware_tflops")
	}
	if !strings.Contains(err.Error(), "hardware_tflops") {
		t.Errorf("error should mention hardware_tflops: %v", err)
	}
}
