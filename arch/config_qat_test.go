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
