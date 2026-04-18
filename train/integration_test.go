//go:build mlx && cgo && darwin

package train

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

// discoverExampleConfigs finds all JSON configs in the examples/ directory.
func discoverExampleConfigs(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir("../examples")
	if err != nil {
		t.Fatalf("ReadDir ../examples: %v", err)
	}
	var configs []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			configs = append(configs, e.Name())
		}
	}
	if len(configs) == 0 {
		t.Fatal("no example JSON configs found")
	}
	return configs
}

// shrinkConfigForTest overrides config dimensions for fast GPU testing:
// small seq_len and batch_tokens=seq_len (batchSize=1), 10 steps.
func shrinkConfigForTest(cfg *arch.ArchConfig) {
	if cfg.SeqLen > 128 {
		cfg.SeqLen = 128
	}
	cfg.Training.Steps = 10
	cfg.Training.BatchTokens = cfg.SeqLen
	cfg.Training.LR = 1e-3
}

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

// TestIntegrationExampleConfigs_TrainStable runs 10 training steps on
// synthetic data for each example config and verifies the training loop
// produces finite, non-NaN losses (no crashes, no gradient explosions).
func TestIntegrationExampleConfigs_TrainStable(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	configs := discoverExampleConfigs(t)

	for _, filename := range configs {
		t.Run(filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			shrinkConfigForTest(cfg)

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

			steps := 10
			for step := 0; step < steps; step++ {
				x, y := generateSyntheticBatch(rng, cfg.Training.BatchTokens, cfg.VocabSize)
				loss, err := trainer.TrainStepGPU(x, y, batchSize, cfg.SeqLen, lr)
				if err != nil {
					t.Fatalf("TrainStepGPU step %d: %v", step, err)
				}
				t.Logf("  step %d/%d loss=%.4f", step, steps, loss)

				if loss != loss { // NaN
					t.Fatalf("loss is NaN at step %d", step)
				}
				if loss <= 0 {
					t.Fatalf("loss is non-positive (%.4f) at step %d", loss, step)
				}
				if loss > 100 {
					t.Fatalf("loss exploded (%.4f) at step %d", loss, step)
				}
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

	configs := discoverExampleConfigs(t)

	for _, filename := range configs {
		t.Run(filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", filename))
			if err != nil {
				t.Fatalf("LoadArchConfig: %v", err)
			}

			shrinkConfigForTest(cfg)
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
