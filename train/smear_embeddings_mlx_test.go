//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestSmearEmbeddingsMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "smear_embeddings_mlx",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"smear_embeddings": true,
		"blocks": [{"type": "plain", "heads": 2}],
		"training": {
			"steps": 1,
			"lr": 0.001,
			"seed": 17,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "smear_embeddings_mlx")
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
}
