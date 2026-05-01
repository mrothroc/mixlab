//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestMTPUntieScheduleMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "mtp_untie_schedule",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"tie_embeddings": true,
		"mtp": {
			"n": 3,
			"loss_weights": [1.0, 0.5, 0.25],
			"untie_embed_at_frac": 0.5
		},
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"seed": 13,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "mtp_untie_schedule")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}

	initialProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: false})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(initial): %v", err)
	}
	finalProg, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{RecurrenceActive: true, HeadUntied: true})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig(final): %v", err)
	}
	trainerIface, err := initGPUTrainer(initialProg, cfg, nil, nil)
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
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
		if step == cfg.EffectiveMTPUntieStep() {
			if err := trainer.CopyWeightGPU("head", "embed"); err != nil {
				t.Fatalf("CopyWeightGPU step %d: %v", step, err)
			}
			if err := trainer.SetProgramGPU(finalProg); err != nil {
				t.Fatalf("SetProgramGPU step %d: %v", step, err)
			}
		}
		x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
	}
}
