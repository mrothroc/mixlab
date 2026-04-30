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

func TestParseArchConfig_BlockScopedParallelResidualValid(t *testing.T) {
	data := []byte(`{
		"name": "parallel_residual_scoped",
		"model_dim": 64,
		"vocab_size": 256,
		"seq_len": 32,
		"blocks": [
			{"type": "gated_deltanet", "heads": 4, "d_k": 8},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4, "parallel_residual": true},
			{"type": "swiglu"}
		]
	}`)
	got, err := ParseArchConfig(data, "parallel_scoped")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.ParallelResidual {
		t.Fatal("top-level ParallelResidual=true, want false")
	}
	if got.Blocks[2].ParallelResidual == nil || !*got.Blocks[2].ParallelResidual {
		t.Fatal("blocks[2].parallel_residual not parsed as true")
	}

	roundTrip, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(roundTrip), `"parallel_residual":true`) {
		t.Fatalf("round-trip JSON missing block parallel_residual: %s", roundTrip)
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

func TestParseArchConfig_BlockScopedParallelResidualValidation(t *testing.T) {
	tests := []struct {
		name       string
		blocks     []BlockSpec
		wantErrSub string
	}{
		{
			name: "start type unsupported",
			blocks: []BlockSpec{
				{Type: "swiglu", ParallelResidual: boolPtr(true)},
				{Type: "swiglu"},
			},
			wantErrSub: "blocks[0].type=plain or gated_deltanet",
		},
		{
			name: "missing swiglu follower",
			blocks: []BlockSpec{
				{Type: "gated_deltanet", Heads: 4, DK: 8, ParallelResidual: boolPtr(true)},
			},
			wantErrSub: "followed by swiglu",
		},
		{
			name: "second block cannot start overlapping pair",
			blocks: []BlockSpec{
				{Type: "gated_deltanet", Heads: 4, DK: 8, ParallelResidual: boolPtr(true)},
				{Type: "swiglu", ParallelResidual: boolPtr(true)},
			},
			wantErrSub: "overlaps pair",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := parallelResidualTestConfig(tc.blocks)
			cfg.ParallelResidual = false
			data, err := json.Marshal(cfg)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			_, err = ParseArchConfig(data, "parallel_scoped_invalid")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErrSub)
			}
		})
	}
}
