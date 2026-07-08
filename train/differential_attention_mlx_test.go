//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDifferentialAttentionTinyTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	trainDir := filepath.Join(dir, "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", trainDir, err)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), []uint16{
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
	})
	trainPattern := filepath.Join(trainDir, "train_*.bin")
	for _, objective := range []string{"causal", "mntp"} {
		t.Run(objective, func(t *testing.T) {
			cfg := differentialAttentionSmokeConfig(objective)
			result, err := runTrain(cfg, trainPattern, TrainOptions{LogEvery: 0, ValEvery: 0})
			if err != nil {
				if strings.Contains(err.Error(), "MLX backend unavailable") {
					t.Skipf("MLX backend not available: %v", err)
				}
				t.Fatalf("runTrain: %v", err)
			}
			if math.IsNaN(result.LastLoss) || math.IsInf(result.LastLoss, 0) {
				t.Fatalf("last loss is non-finite: %g", result.LastLoss)
			}
		})
	}
}

func differentialAttentionSmokeConfig(objective string) *ArchConfig {
	training := DefaultTrainingSpec()
	training.Steps = 3
	training.LR = 1e-4
	training.BatchTokens = 8
	training.Seed = 77
	training.Objective = objective
	training.MLMMaskTokenID = 1
	training.MLMMaskProb = 0.5
	training.GradClip = 1.0
	training.WeightDecay = 0
	training.EmbedWeightDecay = 0
	training.MatrixWeightDecay = 0
	training.ScalarWeightDecay = 0
	training.HeadWeightDecay = 0
	cfg := &ArchConfig{
		Name:      "diff-attn-smoke",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		MLPMult:   1,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 1, DifferentialAttention: true},
		},
		Training: training,
	}
	return cfg
}
