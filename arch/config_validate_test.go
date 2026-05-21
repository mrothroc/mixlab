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
