package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFMaskedLMMetadataAndBidirectionalBlocks(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, `{
		"name": "hf_masked_lm",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"batch_tokens": 4,
			"seed": 606,
			"objective": "hybrid",
			"mlm_mask_token_id": 1,
			"hybrid_clm_fraction": 0.25,
			"hybrid_secondary_objective": "mntp"
		}
	}`, scaleHFExportWeightsToTrainedMagnitude)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if got := doc.AutoMap["AutoModelForMaskedLM"]; got != "modeling_mixlab.MixlabForMaskedLM" {
		t.Fatalf("AutoModelForMaskedLM auto_map=%v", got)
	}
	if !containsString(doc.Architectures, "MixlabForMaskedLM") {
		t.Fatalf("architectures=%v missing MixlabForMaskedLM", doc.Architectures)
	}
	if len(doc.MaskedBlocks) != len(doc.Blocks) {
		t.Fatalf("masked_blocks=%d blocks=%d", len(doc.MaskedBlocks), len(doc.Blocks))
	}
	if got := doc.Blocks[0]["attention_mask"]; got != "causal" {
		t.Fatalf("causal block attention_mask=%v, want causal", got)
	}
	if got := doc.MaskedBlocks[0]["attention_mask"]; got != "bidirectional" {
		t.Fatalf("masked block attention_mask=%v, want bidirectional", got)
	}
	assertHFModelingTemplateContains(t, filepath.Join(outDir, "modeling_mixlab.py"),
		"masked_blocks if masked_blocks else config.blocks",
		"def forward_hidden(self, input_ids=None, attention_mask=None):",
		"x = block(x, relative_embeddings, dwa, attention_mask)",
		"super().__init__(config, blocks=config.blocks)",
	)

	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load HF weights: %v", err)
	}
	causalBlocks := decodeHFBlockSpecs(t, doc.Blocks)
	maskedBlocks := decodeHFBlockSpecs(t, doc.MaskedBlocks)
	causalCfg := *cfg
	causalCfg.Blocks = causalBlocks
	maskedCfg := *cfg
	maskedCfg.Blocks = maskedBlocks
	tokens := [][]int{{0, 1, 2, 3}}
	causalLogits := runHFCPUForward(t, &causalCfg, hfWeights, tokens)
	maskedLogits := runHFCPUForward(t, &maskedCfg, hfWeights, tokens)
	if diff := maxAbsDiff3D(causalLogits, maskedLogits); diff <= 1e-6 {
		t.Fatalf("masked and causal logits unexpectedly identical; max diff=%g", diff)
	}
}

func TestExportHFPureMaskedObjectiveRegistersMaskedLM(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_pure_mlm",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"seed": 707,
			"objective": "mlm",
			"mlm_mask_token_id": 1
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

	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if got := doc.AutoMap["AutoModelForMaskedLM"]; got != "modeling_mixlab.MixlabForMaskedLM" {
		t.Fatalf("AutoModelForMaskedLM auto_map=%v", got)
	}
	if len(doc.MaskedBlocks) != 1 {
		t.Fatalf("masked_blocks=%d, want 1", len(doc.MaskedBlocks))
	}
	if got := doc.MaskedBlocks[0]["attention_mask"]; got != "bidirectional" {
		t.Fatalf("masked block attention_mask=%v, want bidirectional", got)
	}
}

func TestExportHFBERTMLMHeadMetadataAndWeights(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_bert_mlm_head",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"tie_embeddings": true,
		"mlm_head": "bert",
		"dropout": 0.1,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 4,
			"seed": 717,
			"objective": "mlm",
			"mlm_mask_token_id": 1
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

	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if doc.MLMHead != "bert" {
		t.Fatalf("config mlm_head=%q, want bert", doc.MLMHead)
	}
	if doc.HiddenDropout != 0.1 {
		t.Fatalf("config hidden_dropout=%g, want 0.1", doc.HiddenDropout)
	}
	if !containsAnyString(doc.Mixlab["supported_blocks"], "mlm_head=bert") {
		t.Fatalf("supported_blocks missing mlm_head=bert: %#v", doc.Mixlab["supported_blocks"])
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	for _, name := range []string{
		"lm_head_weight",
		"mlm_head_dense.weight",
		"mlm_head_dense.bias",
		"mlm_head_output_bias",
	} {
		if !hfMappingContains(mapping, name) {
			t.Fatalf("weight_map missing %s", name)
		}
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load HF weights: %v", err)
	}
	for _, name := range []string{
		"lm_head_weight",
		"mlm_head_dense.weight",
		"mlm_head_dense.bias",
		"mlm_head_output_bias",
	} {
		if hfWeights[name] == nil {
			t.Fatalf("model.safetensors missing %s", name)
		}
	}
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	nativeShapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	nativeWeights, err := loadSafetensorsWeights(weightsPath, nativeShapes)
	if err != nil {
		t.Fatalf("load native weights: %v", err)
	}
	tokens := [][]int{{0, 1, 2, 3}}
	nativeLogits := runNativeCPUMaskedForward(t, cfg, nativeWeights, tokens)
	hfLogits := runHFCPUMaskedForward(t, cfg, hfWeights, tokens)
	if diff := maxAbsDiff3D(nativeLogits, hfLogits); diff >= 1e-3 {
		t.Fatalf("BERT MLM head native/export logits max diff=%g, want < 1e-3", diff)
	}
}

func TestExportHFMaskedLMSetsTokenizerMaskToken(t *testing.T) {
	dir := t.TempDir()
	// mlm_mask_token_id=6 maps to vocab token "c" in the fixture tokenizer, so a
	// correct ID-based vocab resolution must surface mask_token "c" (not the unk
	// token), proving the mask token is derived from training.mlm_mask_token_id.
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_mask_token",
		"model_dim": 4,
		"vocab_size": 11,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"seed": 808,
			"objective": "mlm",
			"mlm_mask_token_id": 6
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

	var tokCfg map[string]any
	readJSON(t, filepath.Join(outDir, "tokenizer_config.json"), &tokCfg)
	if got := tokCfg["mask_token"]; got != "c" {
		t.Fatalf("tokenizer_config mask_token=%v, want \"c\"", got)
	}
	if got, ok := tokCfg["mask_token_id"].(float64); !ok || int(got) != 6 {
		t.Fatalf("tokenizer_config mask_token_id=%v, want 6", tokCfg["mask_token_id"])
	}

	var specialMap map[string]any
	readJSON(t, filepath.Join(outDir, "special_tokens_map.json"), &specialMap)
	if got := specialMap["mask_token"]; got != "c" {
		t.Fatalf("special_tokens_map mask_token=%v, want \"c\"", got)
	}
}

func hfMappingContains(mapping []hfWeightMapping, hfName string) bool {
	for _, item := range mapping {
		if item.HF == hfName {
			return true
		}
	}
	return false
}

func containsAnyString(values any, want string) bool {
	list, ok := values.([]any)
	if !ok {
		return false
	}
	for _, item := range list {
		if item == want {
			return true
		}
	}
	return false
}

func TestExportHFCausalOmitsTokenizerMaskToken(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_no_mask_token",
		"model_dim": 4,
		"vocab_size": 11,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 909}
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

	var tokCfg map[string]any
	readJSON(t, filepath.Join(outDir, "tokenizer_config.json"), &tokCfg)
	if _, ok := tokCfg["mask_token"]; ok {
		t.Fatalf("causal export unexpectedly set tokenizer_config mask_token: %#v", tokCfg)
	}
	var specialMap map[string]any
	readJSON(t, filepath.Join(outDir, "special_tokens_map.json"), &specialMap)
	if _, ok := specialMap["mask_token"]; ok {
		t.Fatalf("causal export unexpectedly set special_tokens_map mask_token: %#v", specialMap)
	}
}

func decodeHFBlockSpecs(t *testing.T, blocks []map[string]any) []BlockSpec {
	t.Helper()
	data, err := json.Marshal(blocks)
	if err != nil {
		t.Fatalf("marshal HF blocks: %v", err)
	}
	var specs []BlockSpec
	if err := json.Unmarshal(data, &specs); err != nil {
		t.Fatalf("decode HF blocks: %v", err)
	}
	return specs
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func assertHFModelingTemplateContains(t *testing.T, path string, snippets ...string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	text := string(data)
	for _, snippet := range snippets {
		if !strings.Contains(text, snippet) {
			t.Fatalf("modeling_mixlab.py missing snippet %q", snippet)
		}
	}
}
