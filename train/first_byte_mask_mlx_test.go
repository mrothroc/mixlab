//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestFirstByteMaskTrainingSmokeImprovesUnmaskedBPB(t *testing.T) {
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
	// Include continuation-byte targets so unmasked eval catches regressions
	// outside the first-byte denominator.
	pattern := []byte{0xf0, 0x9f, 0x98, 0x80, 0x9f, 0x98, 0x80, 0x80}
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

	baseBPB := baseResult.LastUnmaskedLoss / math.Log(2)
	maskedBPB := maskedResult.LastUnmaskedLoss / math.Log(2)
	improvement := (baseBPB - maskedBPB) / baseBPB
	t.Logf("baseline unmasked BPB=%.6f masked unmasked BPB=%.6f improvement=%.2f%%", baseBPB, maskedBPB, improvement*100)
	if improvement <= 0.0005 {
		t.Fatalf("masked unmasked-BPB improvement %.2f%%, want >0.05%%", improvement*100)
	}
}

func firstByteMaskSmokeConfig(enabled bool) *ArchConfig {
	const lr = 3e-3
	cfg := &ArchConfig{
		Name:      "first_byte_mask_smoke",
		ModelDim:  16,
		VocabSize: 256,
		SeqLen:    8,
		Blocks: []BlockSpec{
			{Type: "mlp", Activation: "gelu"},
		},
		Training: TrainingSpec{
			Steps:         100,
			LR:            lr,
			Seed:          11,
			BatchTokens:   32,
			FirstByteMask: enabled,
		},
	}
	cfg.Training.ApplyDefaults()
	return cfg
}
