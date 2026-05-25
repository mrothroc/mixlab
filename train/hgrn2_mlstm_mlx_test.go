//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestHGRN2AndMLSTMForwardBackwardShapes(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cases := []struct {
		name  string
		block string
	}{
		{name: "hgrn2", block: `{"type": "hgrn2", "heads": 4, "d_state": 12}`},
		{name: "mlstm", block: `{"type": "mlstm", "heads": 4, "d_k": 8, "d_v": 12}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfgJSON := fmt.Sprintf(`{
				"name": "%s_train_step",
				"model_dim": 48,
				"vocab_size": 96,
				"seq_len": 12,
				"blocks": [
					%s,
					{"type": "swiglu"}
				],
				"training": {"steps": 1, "lr": 5e-4, "seed": 11, "batch_tokens": 24, "grad_clip": 1.0, "weight_decay": 0.01}
			}`, tc.name, tc.block)

			cfg, err := ParseArchConfig([]byte(cfgJSON), tc.name+"_train_step")
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

			trainLoss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
			if err != nil {
				t.Fatalf("TrainStepGPU: %v", err)
			}
			if math.IsNaN(float64(trainLoss)) || math.IsInf(float64(trainLoss), 0) || trainLoss <= 0 {
				t.Fatalf("train loss=%g, want finite positive", trainLoss)
			}
		})
	}
}
