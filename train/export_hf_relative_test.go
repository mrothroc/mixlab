package train

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFDebertaSharedQKReuseParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_deberta_shared_qk_reuse",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "kv_heads": 2, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "relative_attention_parameterization": "shared_qk_reuse", "qk_gain": 1.25},
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "relative_attention_parameterization": "shared_qk_reuse"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 213}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, outDir string) {
		var weightMap []hfWeightMapping
		readJSON(t, filepath.Join(outDir, "weight_map.json"), &weightMap)
		if !containsHFWeight(weightMap, "relative_embeddings") {
			t.Fatalf("weight_map missing root relative_embeddings")
		}
		for _, item := range weightMap {
			if strings.Contains(item.HF, "w_pos_key") || strings.Contains(item.HF, "w_pos_query") || strings.Contains(item.HF, "blocks.0.relative_embeddings") || strings.Contains(item.HF, "blocks.1.relative_embeddings") {
				t.Fatalf("shared_qk_reuse exported unexpected per-block relative tensor %q", item.HF)
			}
		}
		var cfg map[string]any
		readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
		blocks := cfg["blocks"].([]any)
		for i, raw := range blocks {
			block := raw.(map[string]any)
			if got := block["relative_attention_parameterization"]; got != "shared_qk_reuse" {
				t.Fatalf("blocks[%d].relative_attention_parameterization=%v want shared_qk_reuse", i, got)
			}
		}
	})
}
