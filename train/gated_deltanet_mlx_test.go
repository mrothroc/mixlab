//go:build mlx && cgo && darwin

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestGatedDeltaNetForwardBackwardShapes(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "gdn_train_step",
		"model_dim": 64,
		"vocab_size": 128,
		"seq_len": 16,
		"blocks": [
			{"type": "gated_deltanet", "heads": 4, "d_k": 12},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 5e-4, "seed": 7, "batch_tokens": 32, "grad_clip": 1.0, "weight_decay": 0.01}
	}`), "gdn_train_step")
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

	evalLoss, err := trainer.EvaluateGPU(x, y, batchSize, cfg.SeqLen)
	if err != nil {
		t.Fatalf("EvaluateGPU: %v", err)
	}
	if math.IsNaN(float64(evalLoss)) || math.IsInf(float64(evalLoss), 0) || evalLoss <= 0 {
		t.Fatalf("eval loss=%g, want finite positive", evalLoss)
	}

	hidden, err := trainer.ReadOutput("x_hidden", []int{batchSize, cfg.SeqLen, cfg.ModelDim})
	if err != nil {
		t.Fatalf("ReadOutput(x_hidden): %v", err)
	}
	if len(hidden) != batchSize*cfg.SeqLen*cfg.ModelDim {
		t.Fatalf("x_hidden elems=%d, want %d", len(hidden), batchSize*cfg.SeqLen*cfg.ModelDim)
	}

	logits, err := trainer.ReadOutput("logits", []int{batchSize * cfg.SeqLen, cfg.VocabSize})
	if err != nil {
		t.Fatalf("ReadOutput(logits): %v", err)
	}
	if len(logits) != batchSize*cfg.SeqLen*cfg.VocabSize {
		t.Fatalf("logits elems=%d, want %d", len(logits), batchSize*cfg.SeqLen*cfg.VocabSize)
	}

	trainLoss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
	if err != nil {
		t.Fatalf("TrainStepGPU: %v", err)
	}
	if math.IsNaN(float64(trainLoss)) || math.IsInf(float64(trainLoss), 0) || trainLoss <= 0 {
		t.Fatalf("train loss=%g, want finite positive", trainLoss)
	}
}
