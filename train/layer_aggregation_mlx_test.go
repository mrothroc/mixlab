//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

// TestLayerAggregationDWAMLXSmoke runs a DWA forward+backward through MLX. DWA
// is the first feature whose aggregation slices the [1, n] materialized alpha
// weights, so this guards the axis/broadcast wiring that pure-CPU IR tests and
// the (HF_PARITY-gated) native-vs-Python parity do not exercise in the default
// MLX suite.
func TestLayerAggregationDWAMLXSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "dwa_mlx_smoke",
		"model_dim": 24,
		"vocab_size": 64,
		"seq_len": 4,
		"layer_aggregation": "dwa",
		"blocks": [
			{"type": "plain", "heads": 3},
			{"type": "swiglu"},
			{"type": "plain", "heads": 3},
			{"type": "mlp", "activation": "gelu"}
		],
		"training": {"steps": 4, "lr": 0.001, "seed": 71, "batch_tokens": 16, "grad_clip": 1.0}
	}`), "dwa_mlx_smoke")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainer, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	defer trainer.CloseTrainer()

	batchSize := cfg.Training.BatchTokens / cfg.SeqLen
	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	for step := 0; step < cfg.Training.TotalSteps(); step++ {
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
