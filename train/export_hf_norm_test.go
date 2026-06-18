package train

import (
	"path/filepath"
	"strings"
	"testing"
)

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
