//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

// Run each stage in a separate process, passing the first stage's checkpoint
// to the second. This tests the real loader/augmentation/validation lifecycle.
func TestGridMemoryDatasetProbe(t *testing.T) {
	if os.Getenv("GRID_MEMORY_DATASET_PROBE") != "1" {
		t.Skip("explicit full-resolution GPU acceptance only")
	}
	stage := os.Getenv("GRID_MEMORY_STAGE")
	if stage != "1" && stage != "2" {
		t.Fatal("GRID_MEMORY_STAGE must be 1 or 2")
	}
	config := fmt.Sprintf("../examples/grid_unet_reference/two_channel_stage%s.json", stage)
	ConfigureCUDAGraphLimits(config, "")
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	cfg, err := LoadArchConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.BatchSize = 16
	cfg.Training.Steps = 200
	if raw := os.Getenv("GRID_MEMORY_STEPS"); raw != "" {
		cfg.Training.Steps, err = strconv.Atoi(raw)
		if err != nil || cfg.Training.Steps < 1 {
			t.Fatal("GRID_MEMORY_STEPS must be positive")
		}
	}
	if stage == "2" {
		cfg.Training.InitFrom = os.Getenv("GRID_MEMORY_INIT")
		if cfg.Training.InitFrom == "" {
			t.Fatal("stage 2 requires GRID_MEMORY_INIT stage-1 checkpoint")
		}
	}
	t.Setenv(mlxMemLogEveryEnv, "10")
	manifest := prepareGridMemoryProbeDataset(t)
	result, err := runGridTrain(cfg, manifest, TrainOptions{
		SafetensorsPath: os.Getenv("GRID_MEMORY_SAVE"), CheckpointDir: t.TempDir(),
		CheckpointEvery: 100, ValEvery: 25, LogEvery: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasValLoss || math.IsNaN(result.LastValLoss) || math.IsInf(result.LastValLoss, 0) {
		t.Fatalf("invalid final validation: %+v", result)
	}
	t.Logf("dataset probe stage=%s first_loss=%.9g last_loss=%.9g val=%.9g elapsed=%s", stage, result.FirstLoss, result.LastLoss, result.LastValLoss, result.Elapsed)
}

func prepareGridMemoryProbeDataset(t *testing.T) string {
	t.Helper()
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy required for prepare integration")
	}
	dir := t.TempDir()
	splits := map[string]any{}
	// Nonmultiples of 16 exercise padding in both training and validation.
	for _, split := range []struct {
		name  string
		count int
	}{{"train", 65}, {"val", 19}} {
		const pixels = 256 * 256
		x := make([]float32, split.count*2*pixels)
		y := make([]float32, split.count*pixels)
		m := make([]float32, len(y))
		for n := 0; n < split.count; n++ {
			for j := 0; j < pixels; j++ {
				a := float32((j+n*17)%251) / 250
				b := float32((j/256+n*7)%127) / 126
				x[n*2*pixels+j], x[n*2*pixels+pixels+j] = a, b
				y[n*pixels+j] = .3*a + .2*b
				if (j+n)%7 != 0 {
					m[n*pixels+j] = 1
				}
			}
		}
		xp, yp, mp := split.name+"-x.npy", split.name+"-y.npy", split.name+"-m.npy"
		writeNPYFloat32(t, filepath.Join(dir, xp), []int{split.count, 2, 256, 256}, x)
		writeNPYFloat32(t, filepath.Join(dir, yp), []int{split.count, 1, 256, 256}, y)
		writeNPYFloat32(t, filepath.Join(dir, mp), []int{split.count, 1, 256, 256}, m)
		splits[split.name] = map[string]string{"inputs": xp, "targets": yp, "masks": mp}
	}
	raw, err := json.Marshal(map[string]any{"records_per_shard": 17, "splits": splits})
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(dir, "source.json")
	if err = os.WriteFile(source, raw, 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "prepared")
	if err = runPrepare(PrepareOptions{Input: source, Output: out, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(out, data.DatasetManifestFilename)
}
