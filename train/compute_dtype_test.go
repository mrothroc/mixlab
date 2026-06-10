package train

import (
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGPUComputeDTypeForTraining(t *testing.T) {
	cfg := &ArchConfig{
		Name:      "bf16_supported",
		ModelDim:  32,
		VocabSize: 128,
		SeqLen:    8,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "geglu"},
			{Type: "mlp"},
			{Type: "moe", NumExperts: 2},
		},
		Training: TrainingSpec{ComputeDType: "bf16"},
	}
	got, err := gpuComputeDTypeForTraining(cfg)
	if err != nil {
		t.Fatalf("gpuComputeDTypeForTraining: %v", err)
	}
	if got != gpu.ComputeDTypeBF16 {
		t.Fatalf("compute dtype=%d, want BF16", got)
	}

	cfg.Training.ComputeDType = ""
	got, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		t.Fatalf("gpuComputeDTypeForTraining(default): %v", err)
	}
	if got != gpu.ComputeDTypeFloat32 {
		t.Fatalf("default compute dtype=%d, want Float32", got)
	}
}

func TestValidateMLXComputeDTypeConfigRejectsUnsupportedBF16(t *testing.T) {
	tests := []struct {
		name string
		cfg  ArchConfig
		want string
	}{
		{
			name: "qat",
			cfg: ArchConfig{
				Training: TrainingSpec{ComputeDType: "bf16", QAT: "int8"},
				Blocks:   []BlockSpec{{Type: "plain", Heads: 4}},
			},
			want: "training.qat",
		},
		{
			name: "gated_deltanet",
			cfg: ArchConfig{
				Training: TrainingSpec{ComputeDType: "bf16"},
				Blocks:   []BlockSpec{{Type: "gated_deltanet", Heads: 4, DK: 8}},
			},
			want: "blocks[0].type",
		},
		{
			name: "custom",
			cfg: ArchConfig{
				Training: TrainingSpec{ComputeDType: "bf16"},
				Blocks:   []BlockSpec{{Type: "custom", Name: "x"}},
			},
			want: "blocks[0].type",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMLXComputeDTypeConfig(&tc.cfg)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%q, want mention of %q", err.Error(), tc.want)
			}
		})
	}
}
