package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func parallelResidualTestConfig(blocks []BlockSpec) ArchConfig {
	return ArchConfig{
		Name:             "parallel_residual",
		ModelDim:         64,
		VocabSize:        256,
		SeqLen:           32,
		ParallelResidual: true,
		Blocks:           blocks,
	}
}

func TestParseArchConfig_ParallelResidualValid(t *testing.T) {
	cfg := parallelResidualTestConfig([]BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	})
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "parallel")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if !got.ParallelResidual {
		t.Fatal("ParallelResidual=false, want true")
	}
}

func TestParseArchConfig_ParallelResidualValidation(t *testing.T) {
	tests := []struct {
		name       string
		cfg        ArchConfig
		wantErrSub string
	}{
		{
			name: "odd blocks",
			cfg: parallelResidualTestConfig([]BlockSpec{
				{Type: "plain", Heads: 4},
				{Type: "swiglu"},
				{Type: "plain", Heads: 4},
			}),
			wantErrSub: "even number of blocks",
		},
		{
			name: "first not plain",
			cfg: parallelResidualTestConfig([]BlockSpec{
				{Type: "swiglu"},
				{Type: "swiglu"},
			}),
			wantErrSub: "blocks[0].type=plain",
		},
		{
			name: "second not swiglu",
			cfg: parallelResidualTestConfig([]BlockSpec{
				{Type: "plain", Heads: 4},
				{Type: "plain", Heads: 4},
			}),
			wantErrSub: "blocks[1].type=swiglu",
		},
		{
			name: "unet unsupported",
			cfg: func() ArchConfig {
				cfg := parallelResidualTestConfig([]BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}})
				cfg.UNet = true
				return cfg
			}(),
			wantErrSub: "cannot enable parallel_residual with unet",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data, err := json.Marshal(tc.cfg)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			_, err = ParseArchConfig(data, "parallel")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErrSub)
			}
		})
	}
}
