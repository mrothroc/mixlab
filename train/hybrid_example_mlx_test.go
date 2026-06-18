//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestHybridExampleMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "hybrid_example_mlx_smoke",
		"model_dim": 24,
		"vocab_size": 64,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 3, "attention_mask": "bidirectional"},
			{"type": "geglu"}
		],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"seed": 73,
			"batch_tokens": 16,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "hybrid",
			"hybrid_mix_granularity": "example",
			"hybrid_clm_fraction": 0.5,
			"hybrid_secondary_objective": "mntp",
			"mlm_mask_prob": 0.5,
			"mlm_mask_token_id": 63,
			"mlm_mask_token_prob": 1.0,
			"mlm_random_token_prob": 0.0,
			"mlm_kept_unchanged_prob": 0.0
		}
	}`), "hybrid_example_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		Objective:        arch.ObjectiveHybridExample,
	})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	seenCausal := false
	seenMasked := false
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
		x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
		prepared, err := prepareObjectiveBatch(cfg, trainBatch{x: x, y: y}, step, arch.ObjectiveHybridExample)
		if err != nil {
			t.Fatalf("prepare step %d: %v", step, err)
		}
		for _, causal := range prepared.attentionCausal {
			if causal > 0 {
				seenCausal = true
			} else {
				seenMasked = true
			}
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}
	if !seenCausal || !seenMasked {
		t.Fatalf("hybrid example smoke saw causal=%v masked=%v, want both", seenCausal, seenMasked)
	}
}
