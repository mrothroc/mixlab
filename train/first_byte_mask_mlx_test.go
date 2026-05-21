//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestFirstByteMaskTrainingSmokeImprovesObjectiveBPB(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 100-step MLX training smoke in short mode")
	}
	if !MLXAvailable() {
		t.Skip("MLX backend not available")
	}

	dir := t.TempDir()
	dataDir := filepath.Join(dir, "data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(data): %v", err)
	}
	tokens := make([]uint16, 4096)
	pattern := []byte("the quick brown fox jumps over the lazy dog. ")
	for i := range tokens {
		tokens[i] = uint16(pattern[i%len(pattern)])
	}
	writeInferenceShard(t, filepath.Join(dataDir, "train_000.bin"), tokens)
	trainPattern := filepath.Join(dataDir, "train_*.bin")

	base := firstByteMaskSmokeConfig(false)
	baseResult, err := runTrain(base, trainPattern, TrainOptions{})
	if err != nil {
		t.Fatalf("runTrain baseline: %v", err)
	}

	masked := firstByteMaskSmokeConfig(true)
	maskedResult, err := runTrain(masked, trainPattern, TrainOptions{})
	if err != nil {
		t.Fatalf("runTrain masked: %v", err)
	}

	baseBPB := baseResult.LastLoss / math.Log(2)
	maskedBPB := maskedResult.LastLoss / math.Log(2)
	improvement := (baseBPB - maskedBPB) / baseBPB
	t.Logf("baseline objective BPB=%.6f masked objective BPB=%.6f improvement=%.2f%%", baseBPB, maskedBPB, improvement*100)
	if improvement <= 0.05 {
		t.Fatalf("masked objective BPB improvement %.2f%%, want >5%%", improvement*100)
	}
}

func firstByteMaskSmokeConfig(enabled bool) *ArchConfig {
	cfg := &ArchConfig{
		Name:      "first_byte_mask_smoke",
		ModelDim:  16,
		VocabSize: 256,
		SeqLen:    8,
		Blocks: []BlockSpec{
			{Type: "mlp", Activation: "gelu"},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Steps = 100
	cfg.Training.LR = 2e-3
	cfg.Training.Seed = 11
	cfg.Training.BatchTokens = 32
	cfg.Training.WeightDecay = 0
	cfg.Training.FirstByteMask = enabled
	return cfg
}
