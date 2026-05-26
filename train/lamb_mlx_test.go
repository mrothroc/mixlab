//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestLAMBMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "lamb_mlx_smoke",
		"model_dim": 24,
		"vocab_size": 48,
		"seq_len": 6,
		"blocks": [
			{"type": "plain", "heads": 3},
			{"type": "swiglu"}
		],
		"training": {
			"steps": 6,
			"lr": 0.001,
			"seed": 29,
			"batch_tokens": 12,
			"grad_clip": 1.0,
			"weight_decay": 0.01,
			"optimizer": "lamb"
		}
	}`), "lamb_mlx_smoke")
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
	var firstLoss, lastLoss float32
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 {
			t.Fatalf("loss step %d=%g, want finite positive", step, loss)
		}
		if step == 0 {
			firstLoss = loss
		}
		lastLoss = loss
	}
	if lastLoss > firstLoss*1.2 {
		t.Fatalf("loss diverged: first=%g last=%g", firstLoss, lastLoss)
	}
}
