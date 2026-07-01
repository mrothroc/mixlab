package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFMultiheadExportsScorerOnly(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_multihead",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
				 "diffusion": {"block_size": 2, "timestep_conditioning": "adaln", "timestep_conditioning_dim": 5}}
			]
		}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if hfCfg.MLMHead != "bert" {
		t.Fatalf("mlm_head=%q, want bert", hfCfg.MLMHead)
	}
	if _, ok := hfCfg.AutoMap["AutoModelForMaskedLM"]; !ok {
		t.Fatalf("AutoModelForMaskedLM missing from auto_map: %+v", hfCfg.AutoMap)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, want := range []string{"final_norm.weight", "mlm_head_dense.weight", "mlm_head_dense.bias", "mlm_head_output_bias", "lm_head_weight"} {
		if !containsHFWeight(mapping, want) {
			t.Fatalf("weight_map missing %q: %+v", want, mapping)
		}
	}
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "denoiser") || strings.Contains(entry.Mixlab, "adaln") || strings.Contains(entry.HF, "denoiser") {
			t.Fatalf("native-only denoiser/AdaLN weight exported: %+v", entry)
		}
	}
}

func TestExportHFMultiheadRejectsHeadLevelDWA(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_multihead_dwa",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"objective": "multihead",
			"steps": 1,
			"lr": 0.001,
			"batch_tokens": 8,
			"mlm_mask_token_id": 31,
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7, "layer_aggregation": "dwa"},
				{"name": "denoiser", "objective": "block_diffusion", "loss_weight": 0.3,
				 "diffusion": {"block_size": 2}}
			]
		}
	}`)
	err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: filepath.Join(dir, "hf_out"), TokenizerSource: tokenizerDir})
	if err == nil || !strings.Contains(err.Error(), "training.export_head.layer_aggregation") {
		t.Fatalf("RunExportHF error=%v, want layer_aggregation rejection", err)
	}
}

func readJSONFileForTest(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}
