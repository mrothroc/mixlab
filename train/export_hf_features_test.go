package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFData2VecStripsTrainingOnlyPredictor(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_d2v",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"seed": 789,
			"objective": "hybrid",
			"mlm_mask_token_id": 1,
			"hybrid_clm_fraction": 0.5,
			"hybrid_secondary_objective": "mntp",
			"data2vec": {"top_k_layers": 1}
		}
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
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "data2vec") || strings.Contains(entry.HF, "data2vec") {
			t.Fatalf("exported data2vec training-only weight mapping: %#v", entry)
		}
	}

	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	for name := range tensors {
		if strings.Contains(name, "data2vec") {
			t.Fatalf("exported data2vec training-only tensor %q", name)
		}
	}
	if _, ok := tensors["lm_head_weight"]; !ok {
		t.Fatal("exported data2vec+tied config did not materialize lm_head_weight")
	}

	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if _, ok := cfg["training"]; ok {
		t.Fatalf("exported config unexpectedly contains training config: %#v", cfg["training"])
	}
	if _, ok := cfg["data2vec"]; ok {
		t.Fatalf("exported config unexpectedly contains data2vec: %#v", cfg["data2vec"])
	}
}

func TestExportHFDistillationStripsTrainingOnlyConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_distilled_student",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"seed": 789,
			"objective": "causal",
			"distillation": {
				"teacher_checkpoints": ["runs/teacher.safetensors"],
				"teacher_configs": ["configs/teacher.json"],
				"loss_weight_ce": 0.5,
				"loss_weight_kl": 0.5,
				"temperature": 2.0
			}
		}
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

	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if _, ok := cfg["training"]; ok {
		t.Fatalf("exported config unexpectedly contains training config: %#v", cfg["training"])
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "distill") || strings.Contains(entry.HF, "distill") {
			t.Fatalf("exported distillation training-only weight mapping: %#v", entry)
		}
	}
}

func TestExportHFPLLMarginStripsTrainingOnlyConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_pll_margin",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"batch_tokens": 6,
			"seed": 789,
			"objective": "mlm",
			"mlm_mask_token_id": 1,
			"pll_margin": {"path": "pairs.bin", "weight": 1.0}
		}
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
	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if _, ok := cfg["training"]; ok {
		t.Fatalf("exported config unexpectedly contains training config: %#v", cfg["training"])
	}
	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, entry := range mapping {
		if strings.Contains(entry.Mixlab, "pll_margin") || strings.Contains(entry.HF, "pll_margin") {
			t.Fatalf("exported PLL-margin training-only mapping: %#v", entry)
		}
	}
}

func TestExportHFDifferentialAttentionConfigAndWeightMap(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_diff_attention",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "plain", "heads": 1, "differential_attention": true, "differential_lambda_init": 0.25},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 789, "objective": "causal"}
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
	if got := cfg.Blocks[0]["differential_attention"]; got != true {
		t.Fatalf("differential_attention=%v want true", got)
	}
	if got := cfg.Blocks[0]["differential_lambda_init"]; got != 0.25 {
		t.Fatalf("differential_lambda_init=%v want 0.25", got)
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	want := map[string]bool{
		"blocks.0.diff_lambda_q1":    false,
		"blocks.0.diff_lambda_k1":    false,
		"blocks.0.diff_lambda_q2":    false,
		"blocks.0.diff_lambda_k2":    false,
		"blocks.0.diff_subln.weight": false,
	}
	for _, entry := range mapping {
		if _, ok := want[entry.HF]; ok {
			want[entry.HF] = true
		}
	}
	for name, seen := range want {
		if !seen {
			t.Fatalf("missing HF mapping for %s in %#v", name, mapping)
		}
	}
}

func TestHFConfigWritesHalfRotationRoPEConvention(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "hf_rope_convention",
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2, "rope_convention": "half_rotation"},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 8}
	}`), "hf_rope_convention")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	path := filepath.Join(t.TempDir(), "config.json")
	if err := writeHFConfig(path, cfg, hfTokenizerSpecials{}); err != nil {
		t.Fatalf("writeHFConfig: %v", err)
	}
	var doc hfConfigJSON
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}
	if got := doc.Blocks[0]["rope_convention"]; got != "half_rotation" {
		t.Fatalf("rope_convention=%v, want half_rotation", got)
	}

	cfg.Blocks[0].RopeConvention = ""
	path = filepath.Join(t.TempDir(), "config.json")
	if err := writeHFConfig(path, cfg, hfTokenizerSpecials{}); err != nil {
		t.Fatalf("writeHFConfig default: %v", err)
	}
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatalf("read default config: %v", err)
	}
	doc = hfConfigJSON{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("unmarshal default config: %v", err)
	}
	if _, ok := doc.Blocks[0]["rope_convention"]; ok {
		t.Fatalf("default adjacent rope_convention should be omitted, got %v", doc.Blocks[0]["rope_convention"])
	}
}

func TestExportHFWritesXSASparseGateConfigAndWeightMap(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_xsa_sparse_gate_config",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "xsa": true, "sparse_attn_gate": true}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 215}
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
	if got := cfg.Blocks[0]["xsa"]; got != true {
		t.Fatalf("xsa=%v, want true", got)
	}
	if got := cfg.Blocks[0]["sparse_attn_gate"]; got != true {
		t.Fatalf("sparse_attn_gate=%v, want true", got)
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	found := false
	for _, entry := range mapping {
		if entry.HF == "blocks.0.attn_gate_w" {
			found = true
			if len(entry.Shape) != 2 || entry.Shape[0] != 2 || entry.Shape[1] != 8 {
				t.Fatalf("attn_gate_w shape=%v, want [2 8]", entry.Shape)
			}
		}
	}
	if !found {
		t.Fatal("weight_map missing blocks.0.attn_gate_w")
	}
}
