//go:build mlx && cgo && (darwin || linux)

package train

import "testing"

func TestGenerateDiffusionMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "generate_diffusion_mlx_smoke",
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
			"seed": 41,
			"batch_tokens": 4,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 15,
			"diffusion": {
				"block_size": 2,
				"steps_per_block": 2,
				"min_mask_fraction": 1.0,
				"max_mask_fraction": 1.0,
				"confidence_threshold": 0.0,
				"commit_floor": 2
			}
		}
	}`), "generate_diffusion_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := buildGenerateDiffusionIRProgram(cfg)
	if err != nil {
		t.Fatalf("buildGenerateDiffusionIRProgram: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()
	evaluator, ok := trainer.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer type=%T does not implement diffusionGenerationEvaluator", trainer)
	}

	result, err := generateDiffusionTokens(cfg, evaluator, []int{1}, 2)
	if err != nil {
		t.Fatalf("generateDiffusionTokens: %v", err)
	}
	if len(result.tokens) != 3 {
		t.Fatalf("generated token length = %d, want 3 (%v)", len(result.tokens), result.tokens)
	}
	for i, token := range result.tokens {
		if token < 0 || token >= cfg.VocabSize {
			t.Fatalf("token[%d]=%d out of range [0,%d)", i, token, cfg.VocabSize)
		}
	}
}
