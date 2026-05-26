//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestDebertaRelativeAttentionCausalAndMLMSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	t.Run("causal", func(t *testing.T) {
		cfg, err := ParseArchConfig([]byte(`{
			"name": "deberta_relative_causal_smoke",
			"model_dim": 32,
			"vocab_size": 64,
			"seq_len": 8,
			"blocks": [
				{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
				{"type": "swiglu"}
			],
			"training": {"steps": 1, "lr": 0.001, "seed": 17, "batch_tokens": 16, "grad_clip": 1.0, "weight_decay": 0.0}
		}`), "deberta_relative_causal_smoke")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		prog, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatalf("BuildIRProgramFromConfig: %v", err)
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
		x, y := generateSyntheticBatch(rand.New(rand.NewSource(cfg.Training.Seed)), cfg.Training.BatchTokens, cfg.VocabSize)
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU: %v", err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss=%g, want finite positive", loss)
		}
	})

	t.Run("mlm", func(t *testing.T) {
		cfg, err := ParseArchConfig([]byte(`{
			"name": "deberta_relative_mlm_smoke",
			"model_dim": 32,
			"vocab_size": 64,
			"seq_len": 8,
			"blocks": [
				{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 4},
				{"type": "swiglu"}
			],
			"training": {
				"steps": 1,
				"lr": 0.001,
				"seed": 23,
				"batch_tokens": 16,
				"grad_clip": 1.0,
				"weight_decay": 0.0,
				"objective": "mlm",
				"mlm_mask_prob": 0.5,
				"mlm_mask_token_id": 63,
				"mlm_mask_token_prob": 1.0,
				"mlm_random_token_prob": 0.0,
				"mlm_kept_unchanged_prob": 0.0
			}
		}`), "deberta_relative_mlm_smoke")
		if err != nil {
			t.Fatalf("ParseArchConfig: %v", err)
		}
		prog, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatalf("BuildIRProgramFromConfig: %v", err)
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
		x, y := generateSyntheticBatch(rand.New(rand.NewSource(cfg.Training.Seed)), cfg.Training.BatchTokens, cfg.VocabSize)
		prepared, err := prepareObjectiveBatch(cfg, trainBatch{x: x, y: y}, 0, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepareObjectiveBatch: %v", err)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU: %v", err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss=%g, want finite positive", loss)
		}
	})
}
