//go:build mlx && cgo && (darwin || linux)

package train

import (
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
		name   string
		config string
	}{
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
			runNativePythonParityCase(t, python, script, tc.config)
		})
	}
}

func runNativePythonParityCase(t *testing.T, python, script, config string) {
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

	nativeLogits := nativeMLXLogitsForParity(t, cfgPath, weightsPath, window, seqLen, vocab)

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

	cmd := exec.Command(python, script, "--dir", outDir)
	out, err := cmd.CombinedOutput()
	t.Logf("hf_parity_check.py output:\n%s", out)
	if err != nil {
		t.Fatalf("native-vs-HF parity failed: %v", err)
	}
}

// nativeMLXLogitsForParity returns row-major [seqLen*vocab] logits from the
// native MLX eval forward for the single sequence window[:seqLen].
func nativeMLXLogitsForParity(t *testing.T, cfgPath, weightsPath string, window []uint16, seqLen, vocab int) []float32 {
	t.Helper()
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
