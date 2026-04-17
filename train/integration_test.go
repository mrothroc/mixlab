//go:build mlx && cgo && darwin

package train

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// generateSyntheticBatch creates random token sequences for training.
// Tokens are uniform random in [0, vocabSize). Targets are the next token
// (shifted by one position within the same random sequence).
func generateSyntheticBatch(rng *rand.Rand, batchTokens, vocabSize int) (x, y []int) {
	// Generate batchTokens+1 random tokens, then x = tokens[:-1], y = tokens[1:]
	raw := make([]int, batchTokens+1)
	for i := range raw {
		raw[i] = rng.Intn(vocabSize)
	}
	x = make([]int, batchTokens)
	y = make([]int, batchTokens)
	copy(x, raw[:batchTokens])
	copy(y, raw[1:batchTokens+1])
	return x, y
}

// TestIntegrationExampleConfigs_TrainDecreasingLoss runs 10 training steps
// on synthetic data for each example config and verifies loss decreases.
func TestIntegrationExampleConfigs_TrainDecreasingLoss(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cases := []struct {
		filename string
	}{
		{"plain_3L.json"},
	}

	for _, tc := range cases {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			// Override training to use small batch for fast testing.
			// Use batch_tokens = seq_len so batchSize=1.
			cfg.Training.Steps = 10
			cfg.Training.BatchTokens = cfg.SeqLen
			cfg.Training.LR = 1e-3 // slightly higher LR for fast convergence on random data

			batchSize := cfg.Training.BatchTokens / cfg.SeqLen
			if batchSize <= 0 {
				batchSize = 1
			}

			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}

			trainer, err := initGPUTrainer(prog, cfg, nil)
			if err != nil {
				t.Fatalf("initGPUTrainer: %v", err)
			}
			defer trainer.CloseTrainer()

			rng := rand.New(rand.NewSource(cfg.Training.Seed))
			lr := float32(cfg.Training.LR)

			var firstLoss, lastLoss float32
			steps := 10
			for step := 0; step < steps; step++ {
				x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
				loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, lr)
				if err != nil {
					t.Fatalf("TrainStepGPU step %d: %v", step, err)
				}
				if step == 0 {
					firstLoss = loss
				}
				lastLoss = loss
				t.Logf("  step %d/%d loss=%.4f", step, steps, loss)
			}

			t.Logf("  first=%.4f last=%.4f delta=%.4f", firstLoss, lastLoss, lastLoss-firstLoss)

			// Loss should decrease after 10 steps of training.
			// On random data with LR=1e-3, the model should memorize somewhat.
			if lastLoss >= firstLoss {
				t.Errorf("loss did not decrease: first=%.4f last=%.4f", firstLoss, lastLoss)
			}
		})
	}
}

// TestIntegrationExampleConfigs_EvalForwardPass verifies that each config
// can run a forward-only evaluation pass (no gradients) on synthetic data.
func TestIntegrationExampleConfigs_EvalForwardPass(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	cases := []struct {
		filename string
	}{
		{"plain_3L.json"},
	}

	for _, tc := range cases {
		t.Run(tc.filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", tc.filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			cfg.Training.BatchTokens = cfg.SeqLen // batchSize=1
			batchSize := 1

			prog, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatalf("BuildIRProgramFromConfig: %v", err)
			}

			trainer, err := initGPUTrainer(prog, cfg, nil)
			if err != nil {
				t.Fatalf("initGPUTrainer: %v", err)
			}
			defer trainer.CloseTrainer()

			rng := rand.New(rand.NewSource(cfg.Training.Seed))
			x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)

			loss, err := trainer.EvaluateGPU(x, y, batchSize, cfg.SeqLen)
			if err != nil {
				t.Fatalf("EvaluateGPU: %v", err)
			}

			t.Logf("  eval loss=%.4f", loss)

			// Loss should be finite and positive (cross-entropy on random data).
			if loss <= 0 {
				t.Errorf("eval loss should be positive, got %.4f", loss)
			}
			if loss != loss { // NaN check
				t.Error("eval loss is NaN")
			}
		})
	}
}
