package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFTokenizerMissingFailsFast(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, _ := writeHFExportFixture(t, dir, `{
		"name": "hf_tiny",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       filepath.Join(dir, "hf_out"),
		TokenizerSource: filepath.Join(dir, "does-not-exist", "tokenizer.json"),
	})
	if err == nil {
		t.Fatal("RunExportHF succeeded with missing tokenizer")
	}
	if !strings.Contains(err.Error(), "tokenizer source") {
		t.Fatalf("missing tokenizer error = %v", err)
	}
}

func TestExportHFTokenizerSourceSidecarsDerived(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_tiny",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	if err := os.Remove(filepath.Join(tokenizerDir, "tokenizer_config.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(tokenizerDir, "special_tokens_map.json")); err != nil {
		t.Fatal(err)
	}
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: filepath.Join(tokenizerDir, "tokenizer.json"),
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var tokenizerConfig map[string]any
	readJSON(t, filepath.Join(outDir, "tokenizer_config.json"), &tokenizerConfig)
	if tokenizerConfig["tokenizer_class"] != "PreTrainedTokenizerFast" {
		t.Fatalf("derived tokenizer_config.json = %#v", tokenizerConfig)
	}
	if tokenizerConfig["pad_token"] != "<|pad|>" || int(tokenizerConfig["pad_token_id"].(float64)) != 0 {
		t.Fatalf("derived tokenizer_config pad token = %#v", tokenizerConfig)
	}
	var special map[string]any
	readJSON(t, filepath.Join(outDir, "special_tokens_map.json"), &special)
	if special["pad_token"] != "<|pad|>" || special["eos_token"] != "<|eos|>" || special["bos_token"] != "<|bos|>" || special["unk_token"] != "<|unk|>" {
		t.Fatalf("derived special_tokens_map.json = %#v", special)
	}
	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if int(cfg["pad_token_id"].(float64)) != 0 || int(cfg["eos_token_id"].(float64)) != 2 || int(cfg["bos_token_id"].(float64)) != 3 || int(cfg["unk_token_id"].(float64)) != 1 {
		t.Fatalf("config special token ids = %#v", cfg)
	}
}

func TestExportHFTiedEmbeddingsMaterializeLMHead(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_tied",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 456}
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

	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	embed, err := decodeSafetensorFloat32("embed_tokens.weight", []int{7, 4}, tensors)
	if err != nil {
		t.Fatalf("decode embed: %v", err)
	}
	head, err := decodeSafetensorFloat32("lm_head_weight", []int{4, 7}, tensors)
	if err != nil {
		t.Fatalf("decode head: %v", err)
	}
	for v := 0; v < 7; v++ {
		for d := 0; d < 4; d++ {
			if got, want := head[d*7+v], embed[v*4+d]; got != want {
				t.Fatalf("lm_head_weight[%d,%d]=%g want embed[%d,%d]=%g", d, v, got, v, d, want)
			}
		}
	}
}
