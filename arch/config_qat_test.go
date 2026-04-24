package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseArchConfig_QATModes(t *testing.T) {
	tests := []struct {
		name string
		qat  string
		want string
	}{
		{name: "default", qat: "", want: "none"},
		{name: "int8", qat: "int8", want: "int8"},
		{name: "int6", qat: "int6", want: "int6"},
		{name: "normalized", qat: " INT8 ", want: "int8"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := ArchConfig{
				Name:      "test_qat",
				ModelDim:  128,
				VocabSize: 1024,
				SeqLen:    128,
				Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
				Training:  TrainingSpec{Steps: 100, LR: 3e-4, QAT: tt.qat},
			}
			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			got, err := ParseArchConfig(data, "test_qat")
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			if got.Training.QAT != tt.want {
				t.Fatalf("training.qat = %q, want %q", got.Training.QAT, tt.want)
			}
		})
	}
}

func TestQATStart(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_qat_start",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4, QAT: "int6", QATStart: 100},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_qat_start")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Training.QATStart != 100 {
		t.Fatalf("training.qat_start = %d, want 100", got.Training.QATStart)
	}
	if got.Training.QAT != "int6" {
		t.Fatalf("training.qat = %q, want int6", got.Training.QAT)
	}
}

func TestQATStartNegative(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_qat_start_negative",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4, QAT: "int6", QATStart: -1},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	_, err = ParseArchConfig(data, "test_qat_start_negative")
	if err == nil {
		t.Fatal("expected error for negative qat_start")
	}
	if !strings.Contains(err.Error(), "training.qat_start") {
		t.Fatalf("error = %q, want mention of training.qat_start", err.Error())
	}
}

func TestQATStartWithoutQAT(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_qat_start_without_qat",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 2}, {Type: "swiglu"}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4, QATStart: 100},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	_, err = ParseArchConfig(data, "test_qat_start_without_qat")
	if err == nil {
		t.Fatal("expected error for qat_start without qat")
	}
	if !strings.Contains(err.Error(), "training.qat_start") {
		t.Fatalf("error = %q, want mention of training.qat_start", err.Error())
	}
}

func TestParseArchConfig_InvalidQAT(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_invalid_qat",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
		Training:  TrainingSpec{Steps: 100, LR: 3e-4, QAT: "fp8"},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test_invalid_qat")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "training.qat") {
		t.Fatalf("error = %q, want mention of training.qat", err.Error())
	}
}
