//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestScoreDiffusionMLXSmokeAndChunkParity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg := testScoreDiffusionMLXConfig(t, arch.ObjectiveBlockDiffusion, "")
	chunked := newScoreDiffusionMLXEvaluator(t, cfg, 2)
	defer chunked.CloseTrainer()
	chunkedEval, ok := chunked.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer type=%T does not implement diffusionGenerationEvaluator", chunked)
	}
	got, err := scoreDiffusionTokens(cfg, chunkedEval, []int{1, 2, 3, 4}, 0, 2)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens chunked: %v", err)
	}
	if len(got) != 4 {
		t.Fatalf("score len=%d, want 4", len(got))
	}
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("score[%d]=%g, want finite", i, v)
		}
	}

	single := newScoreDiffusionMLXEvaluator(t, cfg, 1)
	defer single.CloseTrainer()
	singleEval, ok := single.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer type=%T does not implement diffusionGenerationEvaluator", single)
	}
	want, err := scoreDiffusionTokens(cfg, singleEval, []int{1, 2, 3, 4}, 0, 1)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens single: %v", err)
	}
	requireFloat64SliceNear(t, got, want, 1e-5)
}

func TestScoreDiffusionHybridBlockDiffusionMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg := testScoreDiffusionMLXConfig(t, arch.ObjectiveHybrid, arch.ObjectiveBlockDiffusion)
	trainer := newScoreDiffusionMLXEvaluator(t, cfg, 2)
	defer trainer.CloseTrainer()
	evaluator, ok := trainer.(diffusionGenerationEvaluator)
	if !ok {
		t.Fatalf("trainer type=%T does not implement diffusionGenerationEvaluator", trainer)
	}
	got, err := scoreDiffusionTokens(cfg, evaluator, []int{1, 2, 3, 4}, 1, 2)
	if err != nil {
		t.Fatalf("scoreDiffusionTokens hybrid: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("score len=%d, want 3", len(got))
	}
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("score[%d]=%g, want finite", i, v)
		}
	}
}

func testScoreDiffusionMLXConfig(t *testing.T, objective, secondary string) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name": "score_diffusion_mlx_smoke",
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
			"seed": 47,
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
	}`), "score_diffusion_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	cfg.Training.Objective = objective
	cfg.Training.HybridSecondaryObjective = secondary
	if objective == arch.ObjectiveHybrid {
		cfg.Training.HybridCLMFraction = 0.5
	}
	return cfg
}

func newScoreDiffusionMLXEvaluator(t *testing.T, cfg *ArchConfig, positionBatch int) GPUTrainer {
	t.Helper()
	local := *cfg
	local.Training.BatchTokens = local.SeqLen * positionBatch
	prog, err := buildGenerateDiffusionIRProgram(&local)
	if err != nil {
		t.Fatalf("buildGenerateDiffusionIRProgram: %v", err)
	}
	trainer, err := initGPUTrainer(prog, &local, nil, nil)
	if err != nil {
		if strings.Contains(err.Error(), "MLX backend unavailable") {
			t.Skip(err.Error())
		}
		t.Fatalf("initGPUTrainer: %v", err)
	}
	return trainer
}
