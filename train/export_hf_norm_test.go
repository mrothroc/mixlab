package train

import (
	"path/filepath"
	"strings"
	"testing"
)

// TestExportHFLayerNormAffinePostNoAttnBias guards against a weight-map
// regression where an over-broad "_bias" skip dropped affine-LayerNorm norm
// biases (e.g. post_attn_norm_bias) for blocks without attn_bias, misaligning
// the exported weight map. The block here has affine LayerNorm in post
// placement and no attn_bias, so its norm biases must still be mapped.
func TestExportHFLayerNormAffinePostNoAttnBias(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_ln_affine_post",
		"model_dim": 8,
		"vocab_size": 17,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"norm_type": "layernorm",
		"norm_placement": "post",
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 457}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, hfName := range []string{"blocks.0.post_attn_norm.bias", "blocks.0.post_ffn_norm.bias", "blocks.1.post_norm.bias"} {
		if !containsHFWeight(mapping, hfName) {
			t.Fatalf("affine-LayerNorm post norm bias %s missing from weight map: %#v", hfName, mapping)
		}
	}
	for _, entry := range mapping {
		if strings.HasSuffix(entry.HF, "wq.bias") || strings.HasSuffix(entry.HF, "wo.bias") {
			t.Fatalf("attn projection bias exported without attn_bias: %#v", entry)
		}
	}
}

func TestExportHFLayerNormNoAffineSandwich(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_ln_no_affine",
		"model_dim": 8,
		"vocab_size": 17,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"norm_type": "layernorm",
		"norm_affine": false,
		"norm_eps": 1e-7,
		"norm_placement": "sandwich",
		"ffn_internal_norm": true,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "geglu"},
			{"type": "mlp", "activation": "gelu"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 456}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	var cfg hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if cfg.NormType != "layernorm" || cfg.NormAffine || cfg.NormEps != 1e-7 || cfg.NormPlacement != "sandwich" || !cfg.FFNInternalNorm {
		t.Fatalf("exported norm config = type:%q affine:%v eps:%g placement:%q internal:%v",
			cfg.NormType, cfg.NormAffine, cfg.NormEps, cfg.NormPlacement, cfg.FFNInternalNorm)
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, entry := range mapping {
		if strings.Contains(entry.HF, "norm") {
			t.Fatalf("no-affine LayerNorm should not export norm tensor mapping: %#v", entry)
		}
	}
	for _, hfName := range []string{"embed_tokens.weight", "lm_head_weight", "blocks.0.wq.weight", "blocks.1.w_gate.weight", "blocks.2.w_down.weight"} {
		if !containsHFWeight(mapping, hfName) {
			t.Fatalf("weight map missing %s: %#v", hfName, mapping)
		}
	}
}
