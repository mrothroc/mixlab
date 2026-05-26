//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestMoEMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "moe_mlx_smoke",
		"model_dim": 24,
		"vocab_size": 48,
		"seq_len": 6,
		"blocks": [
			{"type": "plain", "heads": 3},
			{"type": "moe", "num_experts": 3, "top_k": 2, "expert_block": {"type": "swiglu"}, "load_balance_loss_weight": 0.01}
		],
		"training": {
			"steps": 4,
			"lr": 0.001,
			"seed": 31,
			"batch_tokens": 12,
			"grad_clip": 1.0,
			"weight_decay": 0.0
		}
	}`), "moe_mlx_smoke")
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
	if lastLoss > firstLoss*1.5 {
		t.Fatalf("loss diverged: first=%g last=%g", firstLoss, lastLoss)
	}
	if _, err := trainer.EvaluateGPU(x, y, batchSize, cfg.SeqLen); err != nil {
		t.Fatalf("EvaluateGPU: %v", err)
	}
	aux, err := trainer.ReadOutput("moe_aux_loss", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(moe_aux_loss): %v", err)
	}
	entropy, err := trainer.ReadOutput("moe_router_entropy", []int{1})
	if err != nil {
		t.Fatalf("ReadOutput(moe_router_entropy): %v", err)
	}
	if aux[0] < 0 || math.IsNaN(float64(aux[0])) || math.IsInf(float64(aux[0]), 0) {
		t.Fatalf("moe_aux_loss=%g, want finite non-negative", aux[0])
	}
	if entropy[0] <= 0 || math.IsNaN(float64(entropy[0])) || math.IsInf(float64(entropy[0]), 0) {
		t.Fatalf("moe_router_entropy=%g, want finite positive", entropy[0])
	}
}

func TestMoEParallelResidualMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "moe_parallel_mlx_smoke",
		"model_dim": 24,
		"vocab_size": 48,
		"seq_len": 6,
		"parallel_residual": true,
		"blocks": [
			{"type": "plain", "heads": 3},
			{"type": "moe", "num_experts": 2, "top_k": 1, "expert_block": {"type": "mlp", "activation": "relu"}}
		],
		"training": {"steps": 1, "lr": 0.001, "seed": 37, "batch_tokens": 12, "grad_clip": 1.0, "weight_decay": 0.0}
	}`), "moe_parallel_mlx_smoke")
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
