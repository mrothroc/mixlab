//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestRTDMultiheadExamplesRunTwoMLXSteps(t *testing.T) {
	if !mlxAvailable() || !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	for _, filename := range []string{
		"multihead_mntp_rtd_tiny.json",
		"multihead_mntp_rtd_dedicated_tiny.json",
	} {
		t.Run(filename, func(t *testing.T) {
			cfg, err := LoadArchConfig(filepath.Join("examples", filename))
			if err != nil {
				t.Fatalf("LoadArchConfig(%s): %v", filename, err)
			}
			cfg.Training.Steps = 2
			cfg.Training.WarmupSteps = 0
			cfg.Training.HoldSteps = 0
			cfg.Training.WarmdownSteps = 0

			trainDir := filepath.Join(t.TempDir(), "data")
			if err := os.MkdirAll(trainDir, 0o755); err != nil {
				t.Fatalf("MkdirAll(%s): %v", trainDir, err)
			}
			writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), rtdSmokeTokens(cfg.VocabSize, 4096))

			result, err := runTrain(cfg, filepath.Join(trainDir, "train_*.bin"), TrainOptions{LogEvery: 0, ValEvery: 0})
			if err != nil {
				t.Fatalf("runTrain: %v", err)
			}
			for name, v := range map[string]float64{
				"first": result.FirstLoss,
				"last":  result.LastLoss,
			} {
				if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("%s loss=%g, want finite positive", name, v)
				}
			}
		})
	}
}

func rtdSmokeTokens(vocabSize, n int) []uint16 {
	out := make([]uint16, n)
	span := vocabSize - 2
	if span <= 0 {
		span = 1
	}
	for i := range out {
		out[i] = uint16((i % span) + 2)
	}
	return out
}
