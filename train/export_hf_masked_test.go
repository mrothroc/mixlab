package train

import (
	"encoding/json"
	"path/filepath"
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
