//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"
)

// TestBlockDiffusionMaskShapesMLXForwardOutputs is a composition-level golden
// for the block-diffusion attention mask: it checks that the prefix-plus-block
// visibility rule shows up in the *actual MLX forward logits*, not just in the
// masked score tensor (which TestBlockDiffusionMaskMatchesOracle covers in
// isolation). The reference is computed in-process by perturbing one input
// token and observing which positions' logits move, so there is no checked-in
// golden artifact to drift across MLX versions or hardware.
//
// Config: seq_len=4, block_size=2, active block [0,2). Under the block-diffusion
// mask, an active-block query attends to keys < block_end (the whole block,
// bidirectionally, plus any prefix) and never to positions >= block_end; suffix
// queries are causal. So with active block [0,2):
//   - q=0 (active) sees {0,1}, NOT {2,3}
//   - q=1 (active) sees {0,1}, NOT {2,3}
//   - q=2 (suffix, causal) sees {0,1,2}, NOT {3}
//   - q=3 (suffix, causal) sees {0,1,2,3}
//
// Because logits[q] depend only on the positions in q's attention cone (the only
// cross-position op is the masked attention; norms/FFN/head are per-position),
// perturbing a token must change logits[q] iff q is allowed to attend to it.
func TestBlockDiffusionMaskShapesMLXForwardOutputs(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "block_diffusion_mask_effect_mlx",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 1,
			"lr": 0.01,
			"seed": 71,
			"batch_tokens": 4,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 15,
			"diffusion": {"block_size": 2, "min_mask_fraction": 1.0, "max_mask_fraction": 1.0}
		}
	}`), "block_diffusion_mask_effect_mlx")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	prog, err := buildGenerateDiffusionIRProgram(cfg)
	if err != nil {
		t.Fatalf("buildGenerateDiffusionIRProgram: %v", err)
	}
	// Fixed weights for the whole test: one trainer, never re-initialized, and
	// every forward is gradient-free, so the only thing that varies is the input.
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()
	evaluator, ok := trainer.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer type=%T does not implement diffusionGenerationEvaluator", trainer)
	}

	const (
		seqLen     = 4
		blockStart = 0
		blockEnd   = 2
	)
	vocab := cfg.VocabSize

	forward := func(tokens []int) []float32 {
		t.Helper()
		batch, err := diffusionGenerationBatch(tokens, nil, blockStart, blockEnd, seqLen)
		if err != nil {
			t.Fatalf("diffusionGenerationBatch(%v): %v", tokens, err)
		}
		if _, err := evaluator.EvaluateObjectiveGPU(batch, 1, seqLen); err != nil {
			t.Fatalf("EvaluateObjectiveGPU(%v): %v", tokens, err)
		}
		logits, err := evaluator.ReadOutput("logits", []int{seqLen, vocab})
		if err != nil {
			t.Fatalf("ReadOutput(logits): %v", err)
		}
		if len(logits) != seqLen*vocab {
			t.Fatalf("logits length=%d, want %d", len(logits), seqLen*vocab)
		}
		return logits
	}

	rowMaxAbsDiff := func(a, b []float32, pos int) float64 {
		max := 0.0
		for i := 0; i < vocab; i++ {
			d := math.Abs(float64(a[pos*vocab+i] - b[pos*vocab+i]))
			if d > max {
				max = d
			}
		}
		return max
	}

	base := []int{5, 6, 7, 8}
	perturb := func(pos, tok int) []float32 {
		tokens := append([]int(nil), base...)
		tokens[pos] = tok
		return forward(tokens)
	}

	baseLogits := forward(base)

	// Masked attention scores are set to a large negative constant whose softmax
	// weight underflows to exactly zero, so a non-visible perturbation should
	// leave logits bit-identical; allow a tiny tolerance for safety. A visible
	// perturbation changes embeddings by O(1), so a comfortable floor catches it.
	const (
		unchangedTol = 1e-4
		changedMin   = 1e-3
	)

	// 1. Intra-block bidirectional attention: the active query q=0 must attend
	// *forward* to pos1 (this is the property that distinguishes block diffusion
	// from plain causal attention, where q=0 could not see pos1).
	p1 := perturb(1, 9)
	if d := rowMaxAbsDiff(baseLogits, p1, 0); d <= changedMin {
		t.Fatalf("perturbing in-block pos1 left active query q=0 logits unchanged (max diff %g); intra-block bidirectional attention not observed", d)
	}

	// 2. No future-block leakage: the active block [0,2) must not see pos2.
	p2 := perturb(2, 9)
	for _, q := range []int{0, 1} {
		if d := rowMaxAbsDiff(baseLogits, p2, q); d > unchangedTol {
			t.Fatalf("perturbing future pos2 changed active query q=%d logits (max diff %g); future-token leakage through the block-diffusion mask", q, d)
		}
	}
	if d := rowMaxAbsDiff(baseLogits, p2, 2); d <= changedMin {
		t.Fatalf("perturbing pos2 did not change its own logits (max diff %g); perturbation ineffective, leakage checks would be meaningless", d)
	}

	// 3. Causal suffix: pos3 is visible to none of q=0,1 (active, < block_end) or
	// q=2 (suffix, causal, keys <= 2).
	p3 := perturb(3, 9)
	for _, q := range []int{0, 1, 2} {
		if d := rowMaxAbsDiff(baseLogits, p3, q); d > unchangedTol {
			t.Fatalf("perturbing pos3 changed query q=%d logits (max diff %g); a position attended to one it should not see", q, d)
		}
	}
	if d := rowMaxAbsDiff(baseLogits, p3, 3); d <= changedMin {
		t.Fatalf("perturbing pos3 did not change its own logits (max diff %g); perturbation ineffective", d)
	}
}
