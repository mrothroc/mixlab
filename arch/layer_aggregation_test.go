package arch

import (
	"fmt"
	"strings"
	"testing"
)

func TestParseArchConfig_DWALayerAggregationWeightLayout(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "dwa_layout",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"layer_aggregation": "dwa",
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"}
		],
		"training": {"batch_tokens": 4}
	}`), "dwa_layout")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got := cfg.EffectiveLayerAggregation(); got != "dwa" {
		t.Fatalf("EffectiveLayerAggregation=%q, want dwa", got)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	wantTail := []struct {
		name  string
		shape []int
	}{
		{name: "dwa_alpha_0", shape: []int{2}},
		{name: "dwa_alpha_1", shape: []int{3}},
		{name: "dwa_alpha_2", shape: []int{4}},
	}
	if len(metas) < len(wantTail) {
		t.Fatalf("got %d metas, want at least %d", len(metas), len(wantTail))
	}
	tailStart := len(metas) - len(wantTail)
	for i, want := range wantTail {
		got := metas[tailStart+i]
		if got.Name != want.name {
			t.Fatalf("tail meta %d name=%q, want %q", i, got.Name, want.name)
		}
		if got.InitMode != dwaAlphaInitMode {
			t.Fatalf("%s InitMode=%q, want %q", got.Name, got.InitMode, dwaAlphaInitMode)
		}
		if fmt.Sprint(got.Shape) != fmt.Sprint(want.shape) {
			t.Fatalf("%s shape=%v, want %v", got.Name, got.Shape, want.shape)
		}
	}
}

func TestBuildIRProgramFromConfig_DWAUsesTailAlphaWeights(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "dwa_ir",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"layer_aggregation": "dwa",
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "geglu"}
		],
		"training": {"batch_tokens": 4}
	}`), "dwa_ir")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != len(metas) {
		t.Fatalf("NumWeights=%d, want %d", prog.NumWeights, len(metas))
	}
	tailStart := len(metas) - 3
	seenStatic := false
	seenState := map[string]bool{}
	seenAlpha := map[string]int{}
	for _, op := range prog.Ops {
		if op.Code == OpScalarMul && len(op.Outputs) == 1 && op.Outputs[0] == "dwa_static_embeddings" {
			seenStatic = true
		}
		if len(op.Outputs) == 1 && strings.HasPrefix(op.Outputs[0], "dwa_state_") {
			seenState[op.Outputs[0]] = true
		}
		if op.Code == OpSlice && len(op.Inputs) == 1 && strings.HasPrefix(op.Inputs[0], "w") {
			seenAlpha[op.Inputs[0]]++
		}
	}
	if !seenStatic {
		t.Fatal("missing DWA static embedding snapshot")
	}
	for i := 0; i < 3; i++ {
		state := fmt.Sprintf("dwa_state_%d", i)
		if !seenState[state] {
			t.Fatalf("missing %s snapshot", state)
		}
		weight := fmt.Sprintf("w%d", tailStart+i)
		if seenAlpha[weight] == 0 {
			t.Fatalf("missing slice use of tail DWA alpha weight %s", weight)
		}
	}
}

func TestParseArchConfig_DWALayerAggregationValidation(t *testing.T) {
	base := `{
		"name": "dwa_invalid",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"layer_aggregation": "dwa",
		%s,
		"training": {"batch_tokens": 4}
	}`
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "parallel_residual",
			body:    `"parallel_residual": true, "blocks": [{"type": "plain", "heads": 4}, {"type": "swiglu"}]`,
			wantErr: "parallel_residual",
		},
		{
			name:    "recurrence",
			body:    `"recurrence": [0], "blocks": [{"type": "plain", "heads": 4}]`,
			wantErr: "recurrence",
		},
		{
			name:    "unsupported_block",
			body:    `"blocks": [{"type": "hgrn2", "heads": 4}]`,
			wantErr: "does not support",
		},
		{
			name:    "skip_attention",
			body:    `"blocks": [{"type": "plain", "heads": 4, "skip_attention": true}]`,
			wantErr: "skip_attention",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(fmt.Sprintf(base, tt.body)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("ParseArchConfig error=%v, want containing %q", err, tt.wantErr)
			}
		})
	}
}
