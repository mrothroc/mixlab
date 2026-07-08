//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestExportHFNativePythonParity is the load-bearing FR-1 check: for each
// covered architecture it exports a deterministically scaled trained-magnitude
// fixture, runs the
// *actual* embedded modeling_mixlab.py forward through transformers, and asserts
// it agrees with the *actual* native MLX forward on a byte-identical batch.
// Unlike the Go CPU oracle parity tests, nothing here re-implements the HF math,
// so a future drift between the kernels and the shipped Python template fails
// this test by construction.
//
// Gated on HF_PARITY=1 (set by the hf-parity CI workflow) and skipped when the
// Python HF toolchain is unavailable, so default local/CI runs are unaffected.
func TestExportHFNativePythonParity(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1 to run the native-vs-HF Python parity check")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("python HF dependencies unavailable via %q: %v", python, err)
	}
	script, err := filepath.Abs(filepath.Join("testdata", "hf_parity_check.py"))
	if err != nil {
		t.Fatalf("resolve parity script: %v", err)
	}

	cases := []struct {
		name                string
		config              string
		compareMaskedLogits bool
	}{
		{
			// DIFF Transformer two-softmax plain attention in the normal causal
			// export path.
			name: "differential_attention_causal",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 1, "differential_attention": true},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 9898}
			}`,
		},
		{
			// DIFF Transformer attention in a masked objective export. This
			// writes a native masked-graph logits fixture, so the real
			// AutoModelForMaskedLM path is compared numerically rather than only
			// checked for loadability.
			name: "differential_attention_masked",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 1, "differential_attention": true},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 5,
					"seed": 9998,
					"objective": "mlm",
					"mlm_mask_token_id": 1
				}
			}`,
			compareMaskedLogits: true,
		},
		{
			// Q/K/V/O projection biases plus a GELU value gate on the attention
			// output, including a GQA block, against the Python template.
			name: "attn_bias_value_gate",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 4, "kv_heads": 2, "attn_bias": true, "attn_value_gate": true},
					{"type": "plain", "heads": 2, "attn_bias": true},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 3737}
			}`,
		},
		{
			// Attention post-norm explicitly before the output projection,
			// decoupled from the global (pre) norm placement.
			name: "attn_post_norm_before_outproj",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "attn_post_norm": "before_outproj"},
					{"type": "plain", "heads": 2, "attn_post_norm": "after_outproj"},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 4646}
			}`,
		},
		{
			// Shared DeBERTa relative embeddings with an affine LayerNorm on the
			// shared table before per-block Q/K reuse.
			name: "deberta_shared_qk_reuse_embedding_norm",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4, "relative_attention_parameterization": "shared_qk_reuse", "relative_attention_embedding_norm": "layernorm"},
					{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4, "relative_attention_parameterization": "shared_qk_reuse", "relative_attention_embedding_norm": "layernorm"},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 5, "seed": 4747}
			}`,
		},
		{
			// Dense Weighted Aggregation: each sublayer residual point becomes a
			// learned weighted sum of all prior sublayer outputs + embeddings.
			// Mixed plain (2 points) and FFN blocks (1 point each).
			name: "layer_aggregation_dwa",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"layer_aggregation": "dwa",
				"blocks": [
					{"type": "plain", "heads": 2, "rope_dims": 2},
					{"type": "swiglu"},
					{"type": "plain", "heads": 2},
					{"type": "mlp", "activation": "gelu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 5252}
			}`,
		},
		{
			// Partial RoPE, qk_norm, sigmoid SwiGLU, tanh-approx GELU, full RoPE.
			name: "core_rope_qknorm_glu_mlp",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "rope_dims": 2, "qk_norm": true},
					{"type": "geglu"},
					{"type": "plain", "heads": 2},
					{"type": "mlp", "activation": "gelu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 1234}
			}`,
		},
		{
			// Plain-block gated FFN tails: exercises ffn_activation on the
			// attention block itself rather than standalone GLU blocks.
			name: "plain_gated_ffn_tails",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "rope_dims": 2, "ffn_activation": "geglu"},
					{"type": "plain", "heads": 2, "ffn_activation": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 8181}
			}`,
		},
		{
			// GPT-2-style feature set in the custom-code export: learned absolute
			// position embeddings (no RoPE), affine LayerNorm, per-block attn/FFN
			// biases, a pre-FFN norm, and both exact (gelu) and tanh-approx
			// (gelu_new) GELU FFN tails.
			name: "gpt2_features_learned_absolute_gelu",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "max_positions": 6, "mlp_mult": 2.0,
				"norm_type": "layernorm", "norm_affine": true,
				"positional_embedding": "learned_absolute", "embedding_dropout": 0.1,
				"blocks": [
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu_new", "ffn_pre_norm": true, "ffn_bias": true},
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu", "ffn_pre_norm": true, "ffn_bias": true}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 2727}
			}`,
		},
		{
			// Native GPT-2 export path: load the exported directory through stock
			// GPT2LMHeadModel, with no Mixlab custom Python code, and compare to
			// the native MLX forward. Uses GPT-2's default tanh-approx GELU.
			name: "gpt2_native_gelu_new",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "max_positions": 6, "mlp_mult": 2.0,
				"tie_embeddings": true,
				"norm_type": "layernorm", "norm_affine": true,
				"positional_embedding": "learned_absolute", "embedding_dropout": 0.1,
				"hf_export_format": "gpt2",
				"blocks": [
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu_new", "ffn_pre_norm": true, "ffn_bias": true},
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu_new", "ffn_pre_norm": true, "ffn_bias": true}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 4242}
			}`,
		},
		{
			// Native GPT-2 export with exact GELU, represented by GPT2Config's
			// activation_function="gelu".
			name: "gpt2_native_gelu_exact",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "max_positions": 6, "mlp_mult": 2.0,
				"tie_embeddings": true,
				"norm_type": "layernorm", "norm_affine": true,
				"positional_embedding": "learned_absolute",
				"hf_export_format": "gpt2",
				"blocks": [
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu", "ffn_pre_norm": true, "ffn_bias": true},
					{"type": "plain", "heads": 2, "attention_mask": "causal", "attn_bias": true, "ffn_activation": "gelu", "ffn_pre_norm": true, "ffn_bias": true}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 4343}
			}`,
		},
		{
			// Affine LayerNorm in post placement: exercises LayerNorm scale+bias
			// weights plus post_attn/post_ffn norm slots against the Python template.
			name: "layernorm_affine_post",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"norm_type": "layernorm", "norm_placement": "post",
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "geglu"},
					{"type": "mlp", "activation": "gelu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 5151}
			}`,
		},
		{
			// No-affine LayerNorm, sandwich placement, FFN-internal norm: exercises
			// every norm slot with weight-free LayerNorm ops against the template.
			name: "layernorm_noaffine_sandwich_internal",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"norm_type": "layernorm", "norm_affine": false,
				"norm_placement": "sandwich", "ffn_internal_norm": true,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"},
					{"type": "mlp", "activation": "silu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 6262}
			}`,
		},
		{
			// Half-rotation (GPT-NeoX/Llama) RoPE convention, partial and full.
			name: "rope_half_rotation",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "rope_convention": "half_rotation"},
					{"type": "swiglu"},
					{"type": "plain", "heads": 2, "rope_dims": 2, "rope_convention": "half_rotation"},
					{"type": "geglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 4242}
			}`,
		},
		{
			// Grouped-query attention + per-head qk_gain + sliding causal window.
			name: "gqa_qkgain_window",
			config: `{
				"model_dim": 16, "vocab_size": 13, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 4, "kv_heads": 2, "qk_gain": 4.5, "window_size": 3},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 77}
			}`,
		},
		{
			// DeBERTa disentangled relative attention (no RoPE).
			name: "deberta_relative",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 5, "seed": 91}
			}`,
		},
		{
			// GPT-BERT-style shared DeBERTa relative embeddings with per-block Q/K reuse.
			name: "deberta_shared_qk_reuse",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4, "relative_attention_parameterization": "shared_qk_reuse"},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 5, "seed": 92}
			}`,
		},
		{
			// Hybrid export registers both causal and masked-LM heads. The
			// parity script compares the causal head against native logits and
			// also loads/runs AutoModelForMaskedLM when advertised.
			name: "hybrid_masked_lm_head",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 5,
					"seed": 93,
					"objective": "hybrid",
					"mlm_mask_token_id": 1,
					"hybrid_clm_fraction": 0.5,
					"hybrid_secondary_objective": "mntp"
				}
			}`,
		},
		{
			// Hybrid block-diffusion export intentionally exposes only the
			// causal evaluation view; the diffusion sampler is native-only.
			name: "hybrid_block_diffusion_causal_view",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 4, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 4,
					"seed": 9393,
					"objective": "hybrid",
					"hybrid_clm_fraction": 0.5,
					"hybrid_secondary_objective": "block_diffusion",
					"mlm_mask_token_id": 1,
					"diffusion": {"block_size": 2}
				}
			}`,
		},
		{
			// BERT-style masked-LM transform head: the parity script compares
			// causal logits and loads/runs the real exported AutoModelForMaskedLM
			// path so the generated Python transform stack is exercised.
			name: "bert_mlm_head",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 5, "mlp_mult": 2.0,
				"tie_embeddings": true,
				"mlm_head": "bert",
				"blocks": [
					{"type": "plain", "heads": 2, "attn_bias": true, "attn_value_gate": true},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 5,
					"seed": 9494,
					"objective": "mlm",
					"mlm_mask_token_id": 1
				}
			}`,
		},
		{
			// Multihead export: compare the standard scorer-head HF export
			// against the native eval graph for that export head's single-head
			// inference view, not the raw multihead training config.
			name: "multihead_mntp_diffusion_causal_scorer",
			config: `{
				"name": "multihead_mntp_diffusion_causal_scorer",
				"model_dim": 8, "vocab_size": 11, "seq_len": 4, "mlp_mult": 2.0,
				"tie_embeddings": true,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 4,
					"seed": 9595,
					"objective": "multihead",
					"mlm_mask_token_id": 1,
					"export_head": "mntp",
					"diffusion_head": "diff",
					"heads": [
						{
							"name": "mntp",
							"objective": "mntp",
							"loss_weight": 0.7,
							"output_head": "bert_mlm",
							"tie_embeddings": true,
							"final_norm": true
						},
						{
							"name": "diff",
							"objective": "block_diffusion",
							"loss_weight": 0.3,
							"output_head": "linear",
							"tie_embeddings": false,
							"final_norm": true,
							"diffusion": {"block_size": 2}
						}
					]
				}
				}`,
		},
		{
			// Multihead export with head-scoped DWA: the HF model records trunk
			// residual points without rewriting the stream, then aggregates only
			// the exported scorer head. The native reference below uses the
			// actual multihead head logits output so this cannot silently compare
			// against trunk-wide single-head DWA.
			name: "multihead_mntp_dwa_scorer",
			config: `{
				"name": "multihead_mntp_dwa_scorer",
				"model_dim": 8, "vocab_size": 11, "seq_len": 4, "mlp_mult": 2.0,
				"tie_embeddings": true,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"}
				],
				"training": {
					"steps": 1,
					"batch_tokens": 4,
					"seed": 9696,
					"objective": "multihead",
					"mlm_mask_token_id": 1,
					"export_head": "mntp",
					"diffusion_head": "diff",
					"heads": [
						{
							"name": "mntp",
							"objective": "mntp",
							"loss_weight": 0.7,
							"output_head": "linear",
							"tie_embeddings": true,
							"final_norm": true,
							"layer_aggregation": "dwa"
						},
						{
							"name": "diff",
							"objective": "block_diffusion",
							"loss_weight": 0.3,
							"output_head": "linear",
							"tie_embeddings": false,
							"final_norm": true,
							"diffusion": {"block_size": 2}
						}
					]
				}
				}`,
		},
		{
			// XSA projection and sparse per-head attention output gate.
			name: "xsa_sparse_gate",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2, "rope_dims": 4, "xsa": true, "sparse_attn_gate": true, "qk_gain": 1.25},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 92}
			}`,
		},
		{
			// Sparse MoE: top-k routed experts (geglu) + mlp experts.
			name: "moe_topk",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "moe", "num_experts": 4, "top_k": 2, "expert_block": {"type": "geglu"}},
					{"type": "plain", "heads": 2},
					{"type": "moe", "num_experts": 3, "top_k": 1, "expert_block": {"type": "mlp", "activation": "relu"}}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 55}
			}`,
		},
		{
			// Model-level bigram + trigram hashed feature channels.
			name: "bigram_trigram_features",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"bigram_vocab_size": 17, "trigram_vocab_size": 19,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 33}
			}`,
		},
		{
			// Tokenizer-level char feature channel (with projection to model_dim).
			name: "char_features",
			config: `{
				"model_dim": 8, "vocab_size": 11, "seq_len": 6, "mlp_mult": 2.0,
				"char_vocab_size": 260, "char_dim": 6, "char_max_per_token": 4,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "batch_tokens": 6, "seed": 44}
			}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runNativePythonParityCase(t, python, script, tc.config, tc.compareMaskedLogits)
		})
	}
}

func runNativePythonParityCase(t *testing.T, python, script, config string, compareMaskedLogits bool) {
	t.Helper()
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, config, scaleHFExportWeightsToTrainedMagnitude)

	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	seqLen := cfg.SeqLen
	vocab := cfg.VocabSize

	// Deterministic window of seqLen+1 tokens; the trailing token is only the
	// target for the last position and does not affect the compared logits.
	window := make([]uint16, seqLen+1)
	for i := range window {
		window[i] = uint16((i*7 + 3) % vocab)
	}
	inputIDs := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		inputIDs[i] = int(window[i])
	}

	nativeLogits := nativeMLXLogitsForParity(t, cfgPath, weightsPath, outDir, window, seqLen, vocab)

	if err := writeJSONFile(filepath.Join(outDir, "parity_tokens.json"), map[string]any{
		"input_ids": [][]int{inputIDs},
	}); err != nil {
		t.Fatalf("write parity tokens: %v", err)
	}
	if err := writeJSONFile(filepath.Join(outDir, "parity_native_logits.json"), map[string]any{
		"batch":   1,
		"seq_len": seqLen,
		"vocab":   vocab,
		"logits":  nativeLogits,
	}); err != nil {
		t.Fatalf("write parity native logits: %v", err)
	}
	if compareMaskedLogits {
		nativeMaskedLogits := nativeMLXMaskedLogitsForParity(t, cfgPath, weightsPath, window, seqLen, vocab)
		if err := writeJSONFile(filepath.Join(outDir, "parity_native_masked_logits.json"), map[string]any{
			"batch":   1,
			"seq_len": seqLen,
			"vocab":   vocab,
			"logits":  nativeMaskedLogits,
		}); err != nil {
			t.Fatalf("write parity native masked logits: %v", err)
		}
	}

	cmd := exec.Command(python, script, "--dir", outDir)
	out, err := cmd.CombinedOutput()
	t.Logf("hf_parity_check.py output:\n%s", out)
	if err != nil {
		t.Fatalf("native-vs-HF parity failed: %v", err)
	}
}

func nativeMLXMaskedLogitsForParity(t *testing.T, cfgPath, weightsPath string, window []uint16, seqLen, vocab int) []float32 {
	t.Helper()
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig(masked parity): %v", err)
	}
	objective := cfg.Training.EffectiveObjective()
	if objective == "hybrid" {
		objective = cfg.Training.EffectiveHybridSecondaryObjective()
	}
	if objective != "mlm" && objective != "mntp" {
		t.Fatalf("masked parity objective=%q, want mlm or mntp", objective)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive:       true,
		HeadUntied:             cfg.MTPUntieEnabled(),
		MTPAuxInactive:         true,
		DistillationInactive:   true,
		Data2VecInactive:       true,
		ZLossInactive:          true,
		DropoutInactive:        true,
		SegmentMaskInactive:    true,
		ExampleFramingInactive: true,
		Objective:              objective,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(masked parity): %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes(masked parity): %v", err)
	}
	loadedWeights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("load masked parity safetensors %q: %v", weightsPath, err)
	}
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(masked parity): %v", err)
	}
	defer trainer.CloseTrainer()
	xTok := make([]int, seqLen)
	yTok := make([]int, seqLen)
	lossMask := make([]float32, seqLen)
	for pos := 0; pos < seqLen; pos++ {
		xTok[pos] = int(window[pos])
		yTok[pos] = int(window[pos])
		lossMask[pos] = 1
	}
	batch := objectiveBatch{
		x:         xTok,
		y:         yTok,
		lossMask:  lossMask,
		unmaskedX: append([]int(nil), xTok...),
	}
	if _, err := trainer.EvaluateObjectiveGPU(batch, 1, seqLen); err != nil {
		t.Fatalf("EvaluateObjectiveGPU(masked parity): %v", err)
	}
	logits, err := readTrainerOutput(trainer, "logits", []int{seqLen, vocab})
	if err != nil {
		t.Fatalf("read masked logits: %v", err)
	}
	if len(logits) != seqLen*vocab {
		t.Fatalf("native masked logits length=%d, want %d", len(logits), seqLen*vocab)
	}
	return logits
}

// nativeMLXLogitsForParity returns row-major [seqLen*vocab] logits from the
// native MLX eval forward for the single sequence window[:seqLen].
func nativeMLXLogitsForParity(t *testing.T, cfgPath, weightsPath, hfOutDir string, window []uint16, seqLen, vocab int) []float32 {
	t.Helper()
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	if cfg.Training.MultiheadEnabled() {
		if head := cfg.Training.MultiheadExportHead(); head != nil && head.LayerAggregation == "dwa" {
			return nativeMultiheadExportHeadLogitsForParity(t, cfg, weightsPath, window, seqLen, vocab)
		}
		cfgPath, weightsPath = writeHFExportInferenceFixtureForParity(t, cfg, hfOutDir)
	}
	sess, err := NewInferenceSession(cfgPath, weightsPath)
	if err != nil {
		t.Fatalf("NewInferenceSession: %v", err)
	}
	defer sess.Close()
	logits, err := sess.evalLogitsBatch(window)
	if err != nil {
		t.Fatalf("evalLogitsBatch: %v", err)
	}
	if len(logits) != seqLen*vocab {
		t.Fatalf("native logits length=%d, want %d", len(logits), seqLen*vocab)
	}
	return logits
}

func nativeMultiheadExportHeadLogitsForParity(t *testing.T, cfg *ArchConfig, weightsPath string, window []uint16, seqLen, vocab int) []float32 {
	t.Helper()
	head := cfg.Training.MultiheadExportHead()
	if head == nil {
		t.Fatal("multihead config has no export head")
	}
	rawBatch := cfg.Training.BatchTokens / seqLen
	if rawBatch <= 0 {
		rawBatch = 1
	}
	headCount := len(cfg.Training.Heads)
	if headCount <= 0 {
		t.Fatal("multihead config has no heads")
	}
	rawRows := rawBatch * seqLen
	totalRows := rawBatch * headCount
	totalTokens := totalRows * seqLen
	xTok := make([]int, totalTokens)
	yTok := make([]int, totalTokens)
	lossMask := make([]float32, totalTokens)
	diffusionStart := make([]int32, totalRows)
	diffusionEnd := make([]int32, totalRows)
	for row := 0; row < totalRows; row++ {
		for pos := 0; pos < seqLen; pos++ {
			idx := row*seqLen + pos
			xTok[idx] = int(window[pos])
			yTok[idx] = int(window[pos+1])
			lossMask[idx] = 1
		}
		// A zero-length block leaves the block-diffusion mask in its causal
		// fallback path, matching the exported scorer's next-token view while
		// still exercising the real multihead head-scoped DWA graph.
		diffusionStart[row] = 0
		diffusionEnd[row] = 0
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive:       true,
		HeadUntied:             cfg.MTPUntieEnabled(),
		MTPAuxInactive:         true,
		DistillationInactive:   true,
		Data2VecInactive:       true,
		ZLossInactive:          true,
		DropoutInactive:        true,
		SegmentMaskInactive:    true,
		ExampleFramingInactive: true,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(multihead parity): %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes(multihead parity): %v", err)
	}
	loadedWeights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("load multihead safetensors %q: %v", weightsPath, err)
	}
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(multihead parity): %v", err)
	}
	defer trainer.CloseTrainer()
	batch := objectiveBatch{
		x:                   xTok,
		y:                   yTok,
		lossMask:            lossMask,
		unmaskedX:           append([]int(nil), xTok...),
		diffusionBlockStart: diffusionStart,
		diffusionBlockEnd:   diffusionEnd,
		batchSizeOverride:   totalRows,
	}
	if _, err := trainer.EvaluateObjectiveGPU(batch, rawBatch, seqLen); err != nil {
		t.Fatalf("EvaluateObjectiveGPU(multihead parity): %v", err)
	}
	outputName := "logits"
	logits, err := readTrainerOutput(trainer, outputName, []int{rawRows, vocab})
	if err != nil {
		t.Fatalf("read %s: %v", outputName, err)
	}
	if len(logits) != seqLen*vocab {
		t.Fatalf("native multihead logits length=%d, want %d", len(logits), seqLen*vocab)
	}
	return logits
}

func writeHFExportInferenceFixtureForParity(t *testing.T, cfg *ArchConfig, hfOutDir string) (string, string) {
	t.Helper()
	exportCfg := hfExportInferenceConfig(cfg)
	if exportCfg == nil {
		t.Fatal("hfExportInferenceConfig returned nil")
	}
	dir := filepath.Dir(hfOutDir)
	cfgPath := filepath.Join(dir, "hf_export_inference_config.json")
	if err := writeJSONFile(cfgPath, exportCfg); err != nil {
		t.Fatalf("write export inference config: %v", err)
	}
	weightsPath := filepath.Join(dir, "hf_export_inference_weights.safetensors")
	writeNativeSafetensorsFromHFExportForParity(t, exportCfg, hfOutDir, weightsPath)
	return cfgPath, weightsPath
}

func writeNativeSafetensorsFromHFExportForParity(t *testing.T, cfg *ArchConfig, hfOutDir, outPath string) {
	t.Helper()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes(export inference config): %v", err)
	}
	var mappings []hfWeightMapping
	readJSON(t, filepath.Join(hfOutDir, "weight_map.json"), &mappings)
	byMixlab := make(map[string]hfWeightMapping, len(mappings))
	for _, m := range mappings {
		byMixlab[m.Mixlab] = m
	}
	tensors, err := loadSafetensors(filepath.Join(hfOutDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		mixlabName := fmt.Sprintf("w%d_%s", i, shape.Name)
		mapping, ok := byMixlab[mixlabName]
		if !ok {
			t.Fatalf("exported HF weight map missing %q", mixlabName)
		}
		if !intSlicesEqual(mapping.Shape, shape.Shape) {
			t.Fatalf("exported HF weight %q shape=%v, want %v", mixlabName, mapping.Shape, shape.Shape)
		}
		weight, err := decodeSafetensorFloat32(mapping.HF, shape.Shape, tensors)
		if err != nil {
			t.Fatalf("decode exported HF tensor for %q (%s): %v", mixlabName, mapping.HF, err)
		}
		weights[i] = weight
	}
	if err := exportSafetensors(outPath, cfg, shapes, weights); err != nil {
		t.Fatalf("export native safetensors for parity: %v", err)
	}
}
