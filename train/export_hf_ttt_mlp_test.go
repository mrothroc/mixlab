package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFTTTMLPTemplateHasStatelessDualFastPath(t *testing.T) {
	sourceBytes, err := os.ReadFile(filepath.Join("hf_templates", "ttt_mlp_mixlab.py"))
	if err != nil {
		t.Fatalf("read TTT HF template: %v", err)
	}
	source := string(sourceBytes)
	for _, want := range []string{
		"def _stateless_dual_scan",
		"if state is None and not use_cache:",
		"scan = self._stateless_dual_scan(qk, value, lr_logits)",
		"F.conv1d(",
		"eta_lower = torch.tril(eta)",
		"self.gradient_checkpointing = True",
		"scan = torch_checkpoint(",
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("TTT HF template missing %q", want)
		}
	}
	if !strings.Contains(source, "def _online_segment") {
		t.Fatal("TTT HF template no longer retains the cached online recurrence")
	}
	modelingBytes, err := os.ReadFile(filepath.Join("hf_templates", "modeling_mixlab.py"))
	if err != nil {
		t.Fatalf("read HF modeling template: %v", err)
	}
	if !strings.Contains(string(modelingBytes), "supports_gradient_checkpointing = True") {
		t.Fatal("HF modeling template does not expose standard gradient-checkpointing controls")
	}
}

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
