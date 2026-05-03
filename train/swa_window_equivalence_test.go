//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestSWAWindowEqualsSeqLenMatchesFullCausal(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	const (
		modelDim    = 32
		vocabSize   = 64
		seqLen      = 8
		batchTokens = 16
		seed        = 17
	)
	batchSize := batchTokens / seqLen

	buildCfg := func(windowSize int) *ArchConfig {
		t.Helper()
		raw := fmt.Sprintf(`{
			"name": "swa_window_equivalence_%d",
			"model_dim": %d,
			"vocab_size": %d,
			"seq_len": %d,
			"mlp_mult": 2.0,
			"blocks": [
				{"type": "plain", "heads": 4, "window_size": %d},
				{"type": "swiglu"}
			],
			"training": {
				"steps": 1,
				"lr": 3e-4,
				"seed": %d,
				"batch_tokens": %d,
				"grad_clip": 1.0,
				"weight_decay": 0.0
			}
		}`, windowSize, modelDim, vocabSize, seqLen, windowSize, seed, batchTokens)
		cfg, err := ParseArchConfig([]byte(raw), "swa_window_equivalence")
		if err != nil {
			t.Fatalf("ParseArchConfig(window_size=%d): %v", windowSize, err)
		}
		return cfg
	}

	fullCfg := buildCfg(0)
	windowCfg := buildCfg(seqLen)

	fullProg, err := BuildIRProgramFromConfig(fullCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(full): %v", err)
	}
	windowProg, err := BuildIRProgramFromConfig(windowCfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(window): %v", err)
	}

	fullTrainer, err := initGPUTrainer(fullProg, fullCfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(full): %v", err)
	}
	defer fullTrainer.CloseTrainer()

	windowTrainer, err := initGPUTrainer(windowProg, windowCfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer(window): %v", err)
	}
	defer windowTrainer.CloseTrainer()

	x, y := generateSyntheticBatch(rand.New(rand.NewSource(seed+1)), batchTokens, vocabSize)
	fullLoss, err := fullTrainer.EvaluateGPU(x, y, batchSize, seqLen)
	if err != nil {
		t.Fatalf("EvaluateGPU(full): %v", err)
	}
	windowLoss, err := windowTrainer.EvaluateGPU(x, y, batchSize, seqLen)
	if err != nil {
		t.Fatalf("EvaluateGPU(window): %v", err)
	}

	fullLogits, err := readTrainerOutput(fullTrainer, "logits", []int{batchTokens, vocabSize})
	if err != nil {
		t.Fatalf("ReadOutput(full logits): %v", err)
	}
	windowLogits, err := readTrainerOutput(windowTrainer, "logits", []int{batchTokens, vocabSize})
	if err != nil {
		t.Fatalf("ReadOutput(window logits): %v", err)
	}
	if len(fullLogits) != len(windowLogits) {
		t.Fatalf("logit length mismatch: full=%d window=%d", len(fullLogits), len(windowLogits))
	}

	const tol = 1e-6
	if diff := math.Abs(float64(fullLoss - windowLoss)); diff > tol {
		t.Fatalf("loss mismatch: full=%g window=%g diff=%g", fullLoss, windowLoss, diff)
	}
	for i := range fullLogits {
		diff := math.Abs(float64(fullLogits[i] - windowLogits[i]))
		if diff > tol {
			t.Fatalf("logits[%d] mismatch: full=%g window=%g diff=%g", i, fullLogits[i], windowLogits[i], diff)
		}
	}
}
