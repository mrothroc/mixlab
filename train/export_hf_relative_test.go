package train

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestExportHFAttnBiasValueGateSharedRelativeParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_attn_bias_value_gate_shared_relative",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "kv_heads": 2, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "relative_attention_parameterization": "shared_qk_reuse", "attn_bias": true, "attn_value_gate": true, "qk_gain": 1.1}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 211}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, outDir string) {
		var cfg map[string]any
		readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
		blocks, ok := cfg["blocks"].([]any)
		if !ok || len(blocks) != 1 {
			t.Fatalf("config blocks=%#v", cfg["blocks"])
		}
		block, ok := blocks[0].(map[string]any)
		if !ok {
			t.Fatalf("config block=%#v", blocks[0])
		}
		if block["attn_bias"] != true || block["attn_value_gate"] != true {
			t.Fatalf("exported block missing attention flags: %#v", block)
		}
		var mapping []hfWeightMapping
		readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
		for _, name := range []string{"blocks.0.wq.bias", "blocks.0.wk.bias", "blocks.0.wv.bias", "blocks.0.wo.bias"} {
			if !containsHFWeight(mapping, name) {
				t.Fatalf("weight_map missing %s: %#v", name, mapping)
			}
		}
		for _, item := range mapping {
			if item.HF == "blocks.0.wv.weight" && !reflect.DeepEqual(item.Shape, []int{8, 12}) {
				t.Fatalf("wv weight shape=%v want [8 12]", item.Shape)
			}
		}
	})
}

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

func TestExportHFDebertaSharedQKReuseEmbeddingNormParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_deberta_shared_qk_reuse_embedding_norm",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"norm_eps": 1e-6,
		"blocks": [
			{"type": "plain", "heads": 4, "kv_heads": 2, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "relative_attention_parameterization": "shared_qk_reuse", "relative_attention_embedding_norm": "layernorm"},
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "relative_attention_parameterization": "shared_qk_reuse", "relative_attention_embedding_norm": "layernorm"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 217}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, outDir string) {
		var weightMap []hfWeightMapping
		readJSON(t, filepath.Join(outDir, "weight_map.json"), &weightMap)
		for _, name := range []string{"relative_embeddings", "relative_layer_norm.weight", "relative_layer_norm.bias"} {
			if !containsHFWeight(weightMap, name) {
				t.Fatalf("weight_map missing %s: %#v", name, weightMap)
			}
		}
		var cfg map[string]any
		readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
		blocks := cfg["blocks"].([]any)
		for i, raw := range blocks {
			block := raw.(map[string]any)
			if got := block["relative_attention_embedding_norm"]; got != "layernorm" {
				t.Fatalf("blocks[%d].relative_attention_embedding_norm=%v want layernorm", i, got)
			}
		}
	})
}
