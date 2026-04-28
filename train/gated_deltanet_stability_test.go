//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestGatedDeltaNetChunkedTrainStable(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cfg, err := ParseArchConfig([]byte(`{
		"name": "gdn_chunked_stability",
		"model_dim": 128,
		"vocab_size": 256,
		"seq_len": 64,
		"blocks": [
			{"type": "gated_deltanet", "heads": 4, "d_k": 16, "scan_chunk_size": 64},
			{"type": "gated_deltanet", "heads": 4, "d_k": 16, "scan_chunk_size": 64},
			{"type": "gated_deltanet", "heads": 4, "d_k": 16, "scan_chunk_size": 64},
			{"type": "gated_deltanet", "heads": 4, "d_k": 16, "scan_chunk_size": 64}
		],
		"training": {"steps": 1, "lr": 3e-4, "seed": 1337, "batch_tokens": 128, "grad_clip": 1.0, "weight_decay": 0.01}
	}`), "gdn_chunked_stability")
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
	rng := rand.New(rand.NewSource(cfg.Training.Seed))
	x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)

	const steps = 50
	for step := 0; step < steps; step++ {
		loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("TrainStepGPU step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) || loss <= 0 || loss > 100 {
			t.Fatalf("step %d loss=%g, want finite bounded positive", step, loss)
		}

		hidden, err := trainer.ReadOutput("x_hidden", []int{batchSize, cfg.SeqLen, cfg.ModelDim})
		if err != nil {
			t.Fatalf("ReadOutput(x_hidden) step %d: %v", step, err)
		}
		maxAbs := maxFiniteAbs(hidden)
		if math.IsNaN(maxAbs) || math.IsInf(maxAbs, 0) {
			t.Fatalf("step %d x_hidden contains NaN/Inf", step)
		}
		if maxAbs > 1e4 {
			t.Fatalf("step %d x_hidden max abs=%g, want <= 1e4", step, maxAbs)
		}
	}
}

func maxFiniteAbs(vals []float32) float64 {
	maxAbs := 0.0
	for _, v := range vals {
		fv := float64(v)
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			return fv
		}
		absV := math.Abs(fv)
		if absV > maxAbs {
			maxAbs = absV
		}
	}
	return maxAbs
}
