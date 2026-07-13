package train

import (
	"strings"
	"testing"
)

func TestExportHFTTTMLPConfigAndWeightMap(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim": 8,
		"vocab_size": 13,
		"seq_len": 6,
		"mlp_mult": 2.0,
		"blocks": [
			{"type": "ttt_mlp", "heads": 2, "chunk_size": 4, "inner_hidden_mult": 2, "inner_lr_base": 0.05},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 6}
	}`), "ttt_hf")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if err := validateHFExportConfig(cfg); err != nil {
		t.Fatalf("validateHFExportConfig: %v", err)
	}
	blocks := hfBlockEntries(cfg, false)
	if len(blocks) != 2 {
		t.Fatalf("block entries=%d, want 2", len(blocks))
	}
	ttt := blocks[0]
	for key, want := range map[string]any{
		"type": "ttt_mlp", "heads": 2, "chunk_size": 4,
		"inner_hidden_mult": float64(2), "inner_lr_base": float64(0.05),
	} {
		if got := ttt[key]; got != want {
			t.Fatalf("ttt block %s=%v, want %v", key, got, want)
		}
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		weights[i] = make([]float32, shapeProduct(shape.Shape))
	}
	exportShapes, _, err := materializeHFExportWeights(cfg, cfg, shapes, weights)
	if err != nil {
		t.Fatalf("materializeHFExportWeights: %v", err)
	}
	mapping, err := buildHFWeightMap(cfg, exportShapes)
	if err != nil {
		t.Fatalf("buildHFWeightMap: %v", err)
	}
	byMixlab := make(map[string]string, len(mapping))
	for _, item := range mapping {
		byMixlab[item.Mixlab] = item.HF
	}
	wantSuffixes := map[string]string{
		"norm_scale":        "blocks.0.norm.weight",
		"w_qk":              "blocks.0.w_qk.weight",
		"q_conv":            "blocks.0.q_conv_weight",
		"inner_token_coeff": "blocks.0.inner_token_coeff",
		"inner_w1":          "blocks.0.inner_w1",
		"inner_norm_scale":  "blocks.0.inner_norm_scale",
		"post_norm_bias":    "blocks.0.post_norm.bias",
		"w_out":             "blocks.0.w_out.weight",
	}
	for suffix, wantHF := range wantSuffixes {
		found := false
		for mixlab, gotHF := range byMixlab {
			separator := strings.IndexByte(mixlab, '_')
			if separator >= 0 && mixlab[separator+1:] == suffix {
				found = true
				if gotHF != wantHF {
					t.Fatalf("mapping %s=%s, want %s", mixlab, gotHF, wantHF)
				}
			}
		}
		if !found {
			t.Fatalf("missing mapping for TTT weight %s", suffix)
		}
	}
}

func TestExportHFTTTMLPRejectsUncachedMixerComposition(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim": 8, "vocab_size": 13, "seq_len": 4,
		"blocks": [
			{"type": "ttt_mlp", "heads": 2, "chunk_size": 4},
			{"type": "plain", "heads": 2}
		],
		"training": {"steps": 1, "batch_tokens": 4}
	}`), "ttt_hf_mixed")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	err = validateHFExportConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "cache") {
		t.Fatalf("validateHFExportConfig error=%v, want cache composition rejection", err)
	}
}
