package train

import (
	"encoding/json"
	"math"
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

func TestExportHFMultiheadAllowsMinimalPairMLMSpanPLL(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_multihead_pll_ranking",
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
			"export_head": "scorer",
			"minimal_pair": {
				"path": "pairs.jsonl",
				"energy_aggregation": "differing_span",
				"score_source": "mlm_span_pll",
				"score_head": "scorer"
			},
			"heads": [
				{"name": "scorer", "objective": "mntp", "loss_weight": 0.7},
				{"name": "aux", "objective": "causal", "loss_weight": 0.3}
			]
		}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if _, ok := hfCfg.AutoMap["AutoModelForMaskedLM"]; !ok {
		t.Fatalf("AutoModelForMaskedLM missing from auto_map: %+v", hfCfg.AutoMap)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "minimal_pair") || strings.Contains(entry.HF, "minimal_pair") {
			t.Fatalf("minimal-pair training-only mapping exported: %+v", entry)
		}
	}
}

func TestExportHFMultiheadExportsHeadLevelDWA(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, `{
		"name": "hf_multihead_dwa",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "geglu"}
		],
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
	}`, func(weights [][]float32, shapes []WeightShape) error {
		idx := weightShapeIndex(shapes, "head_scorer_dwa_alpha")
		if idx < 0 {
			return os.ErrNotExist
		}
		copy(weights[idx], []float32{0.125, 0.25, 0.375, 0.5})
		return nil
	})
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var hfCfg hfConfigJSON
	readJSONFileForTest(t, filepath.Join(outDir, "config.json"), &hfCfg)
	if hfCfg.LayerAggregation != "dwa" {
		t.Fatalf("layer_aggregation=%q, want dwa", hfCfg.LayerAggregation)
	}
	if hfCfg.LayerAggregationScope != "head" {
		t.Fatalf("layer_aggregation_scope=%q, want head", hfCfg.LayerAggregationScope)
	}
	var mapping []hfWeightMapping
	readJSONFileForTest(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, want := range []string{"dwa_alphas.0", "dwa_alphas.1", "dwa_alphas.2"} {
		if !containsHFWeight(mapping, want) {
			t.Fatalf("weight_map missing %q: %+v", want, mapping)
		}
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("loadHFWeightsForParity: %v", err)
	}
	want := [][]float64{
		{0.125, 0.25},
		{0.125, 0.25, 0.375},
		{0.125, 0.25, 0.375, 0.5},
	}
	names := []string{"dwa_alphas.0", "dwa_alphas.1", "dwa_alphas.2"}
	for i, wantAlpha := range want {
		name := names[i]
		got := hfWeights[name]
		if len(got) != len(wantAlpha) {
			t.Fatalf("%s length=%d want %d", name, len(got), len(wantAlpha))
		}
		for j := range wantAlpha {
			if math.Abs(got[j]-wantAlpha[j]) > 1e-7 {
				t.Fatalf("%s[%d]=%g want %g", name, j, got[j], wantAlpha[j])
			}
		}
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
