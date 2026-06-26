//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestMLMObjectiveLearnsMaskedTokenAboveChance(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "mlm_objective_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 20,
			"lr": 0.05,
			"seed": 19,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "mlm",
			"mlm_mask_prob": 1.0,
			"mlm_mask_token_id": 15,
			"mlm_mask_token_prob": 1.0,
			"mlm_random_token_prob": 0.0,
			"mlm_kept_unchanged_prob": 0.0
		}
	}`), "mlm_objective_smoke")
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
	raw := trainBatch{
		x: []int{3, 3, 3, 3, 3, 3, 3, 3},
		y: []int{0, 0, 0, 0, 0, 0, 0, 0},
	}
	prepared, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	initial, err := evaluatePreparedObjectiveBatch(trainer, prepared, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("initial evaluate: %v", err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		prepared, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveMLM)
		if err != nil {
			t.Fatalf("prepare step %d: %v", step, err)
		}
		if _, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR)); err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
	}
	final, err := evaluatePreparedObjectiveBatch(trainer, prepared, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("final evaluate: %v", err)
	}
	if math.IsNaN(float64(final)) || math.IsInf(float64(final), 0) {
		t.Fatalf("non-finite final loss: %g", final)
	}
	chance := float32(math.Log(float64(cfg.VocabSize)))
	if final >= initial {
		t.Fatalf("final loss = %g, want below initial %g", final, initial)
	}
	if final >= chance {
		t.Fatalf("final loss = %g, want below chance %g", final, chance)
	}
}

func TestBlockDiffusionMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "block_diffusion_mlx_smoke",
		"model_dim": 16,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 2,
			"lr": 0.01,
			"seed": 31,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0,
			"objective": "block_diffusion",
			"mlm_mask_token_id": 15,
			"diffusion": {
				"block_size": 2,
				"min_mask_fraction": 1.0,
				"max_mask_fraction": 1.0
			}
		}
	}`), "block_diffusion_mlx_smoke")
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
	raw := trainBatch{
		x: []int{3, 4, 5, 6, 3, 4, 5, 6},
		y: []int{4, 5, 6, 7, 4, 5, 6, 7},
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		prepared, err := prepareObjectiveBatch(cfg, raw, step, arch.ObjectiveBlockDiffusion)
		if err != nil {
			t.Fatalf("prepare block diffusion step %d: %v", step, err)
		}
		loss, err := trainer.TrainObjectiveStepGPU(prepared, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainObjectiveStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}
}

func evaluatePreparedObjectiveBatch(trainer *mlxGPUTrainer, batch objectiveBatch, batchSize, seqLen int) (float32, error) {
	return trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen)
}
