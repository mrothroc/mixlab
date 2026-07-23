package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
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

func TestExportHFSpecialTokensUseManifestThenExplicitOverrides(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_specials",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	removeTokenizerSpecialDeclarations(t, tokenizerDir)
	writeHFExportDatasetManifest(t, tokenizerDir, 7, map[string]int{
		"pad": 0,
		"bos": 2,
		"eos": 3,
	})
	overrideBOS := 4
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
		BOSTokenID:      &overrideBOS,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if got := int(cfg["pad_token_id"].(float64)); got != 0 {
		t.Fatalf("pad_token_id=%d want manifest id 0", got)
	}
	if got := int(cfg["eos_token_id"].(float64)); got != 3 {
		t.Fatalf("eos_token_id=%d want manifest id 3", got)
	}
	if got := int(cfg["bos_token_id"].(float64)); got != 4 {
		t.Fatalf("bos_token_id=%d want explicit override 4", got)
	}

	var tokenizerConfig map[string]any
	readJSON(t, filepath.Join(outDir, "tokenizer_config.json"), &tokenizerConfig)
	if got := int(tokenizerConfig["bos_token_id"].(float64)); got != 4 {
		t.Fatalf("tokenizer_config bos_token_id=%d want 4", got)
	}
	if got := tokenizerConfig["bos_token"]; got != "a" {
		t.Fatalf("tokenizer_config bos_token=%v want token mapped from id 4", got)
	}
}

func TestExportHFSpecialTokensPreferTokenizerMetadataOverFlags(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_tokenizer_precedence",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	flagBOS := 4
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
		BOSTokenID:      &flagBOS,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if got := int(cfg["bos_token_id"].(float64)); got != 3 {
		t.Fatalf("bos_token_id=%d want tokenizer-declared id 3", got)
	}
}

func TestExportHFSpecialTokensAcceptIDOnlyTokenizerSidecar(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_id_only_specials",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	removeTokenizerSpecialDeclarations(t, tokenizerDir)
	if err := writeJSONFile(filepath.Join(tokenizerDir, "tokenizer_config.json"), map[string]any{
		"tokenizer_class": "PreTrainedTokenizerFast",
		"pad_token_id":    0,
		"bos_token_id":    3,
		"eos_token_id":    2,
	}); err != nil {
		t.Fatal(err)
	}
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
	if int(cfg["pad_token_id"].(float64)) != 0 || int(cfg["bos_token_id"].(float64)) != 3 || int(cfg["eos_token_id"].(float64)) != 2 {
		t.Fatalf("config special token ids = %#v", cfg)
	}
}

func TestExportHFSpecialTokenOverrideRejectsOutOfRangeID(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_specials_bad",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3}
	}`)
	badEOS := 7
	err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       filepath.Join(dir, "hf_out"),
		TokenizerSource: tokenizerDir,
		EOSTokenID:      &badEOS,
	})
	if err == nil || !strings.Contains(err.Error(), "-eos-token-id=7") {
		t.Fatalf("RunExportHF error=%v, want range error naming explicit EOS override", err)
	}
}

func TestExportHFSpecialTokensFallBackToExampleFraming(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_framing_specials",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"objective": "causal",
			"example_framing": {"content_len": 1, "bos_id": 3, "eos_id": 2}
		}
	}`)
	removeTokenizerSpecialDeclarations(t, tokenizerDir)
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
	if got := int(cfg["bos_token_id"].(float64)); got != 3 {
		t.Fatalf("bos_token_id=%d want framing id 3", got)
	}
	if got := int(cfg["eos_token_id"].(float64)); got != 2 {
		t.Fatalf("eos_token_id=%d want framing id 2", got)
	}
}

func removeTokenizerSpecialDeclarations(t *testing.T, tokenizerDir string) {
	t.Helper()
	var tokenizer map[string]any
	readJSON(t, filepath.Join(tokenizerDir, "tokenizer.json"), &tokenizer)
	tokenizer["added_tokens"] = []any{}
	if err := writeJSONFile(filepath.Join(tokenizerDir, "tokenizer.json"), tokenizer); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tokenizer_config.json", "special_tokens_map.json"} {
		if err := os.Remove(filepath.Join(tokenizerDir, name)); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
	}
}

func writeHFExportDatasetManifest(t *testing.T, dir string, vocabSize int, specials map[string]int) {
	t.Helper()
	manifest := data.DatasetManifest{
		Format:          data.DatasetManifestFormat,
		Version:         data.DatasetManifestVersion,
		Representation:  data.DatasetRepresentationDiscreteTokens,
		Modality:        "text",
		VocabSize:       vocabSize,
		TokenDType:      data.DatasetTokenDTypeUint16,
		ShardFormat:     data.DatasetShardFormatSequenceV1,
		SequenceLayout:  data.DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:    3,
		SpecialTokenIDs: specials,
		Artifacts:       data.DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin"},
		},
	}
	if err := writeJSONFile(filepath.Join(dir, data.DatasetManifestFilename), manifest); err != nil {
		t.Fatal(err)
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
