//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"path/filepath"
	"testing"
)

func TestGridMuonTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	for _, optimizer := range []string{"muon", "muon_eq_r", "normuon"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_muon_tiny.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Optimizer = optimizer
			cfg.Training.Steps = 200
			dir := t.TempDir()
			result, err := runGridTrain(cfg, manifest, TrainOptions{SafetensorsPath: filepath.Join(dir, "model.safetensors"), ValEvery: 100, LogEvery: 100})
			if err != nil {
				t.Fatal(err)
			}
			if math.IsNaN(float64(result.LastLoss)) || math.IsInf(float64(result.LastLoss), 0) || math.IsNaN(float64(result.LastValLoss)) || math.IsInf(float64(result.LastValLoss), 0) {
				t.Fatalf("nonfinite result %+v", result)
			}
		})
	}
}
