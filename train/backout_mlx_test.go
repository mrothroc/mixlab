//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestBackoutMLXSmokeUpdatesLambda(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "backout_mlx",
		"model_dim": 16,
		"vocab_size": 32,
		"seq_len": 4,
		"backout": {"save_layer": 1, "lambda_init": -1.0},
		"blocks": [
			{"type": "plain", "heads": 2},
			{"type": "swiglu"},
			{"type": "plain", "heads": 2}
		],
		"training": {
			"steps": 5,
			"lr": 0.001,
			"seed": 19,
			"batch_tokens": 8,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "backout_mlx")
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

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	lambdaIdx := -1
	for i, shape := range shapes {
		if shape.Name == "backout_lambda" {
			lambdaIdx = i
			break
		}
	}
	if lambdaIdx < 0 {
		t.Fatal("missing backout_lambda weight")
	}
	before, err := trainer.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights(before): %v", err)
	}
	initialLambda := before[lambdaIdx][0]
	if initialLambda != -1.0 {
		t.Fatalf("initial backout_lambda=%g want -1", initialLambda)
	}

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	x, y := generateSyntheticBatch(rand.New(rand.NewSource(cfg.Training.Seed)), cfg.Training.BatchTokens, cfg.VocabSize)
	firstLoss := float32(0)
	lastLoss := float32(0)
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
	after, err := trainer.ReadWeights()
	if err != nil {
		t.Fatalf("ReadWeights(after): %v", err)
	}
	finalLambda := after[lambdaIdx][0]
	if math.Abs(float64(finalLambda-initialLambda)) < 1e-7 {
		t.Fatalf("backout_lambda did not change: initial=%g final=%g", initialLambda, finalLambda)
	}
	if lastLoss > firstLoss*1.2 {
		t.Fatalf("loss diverged: first=%g last=%g", firstLoss, lastLoss)
	}
}
