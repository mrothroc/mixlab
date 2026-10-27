//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// Opt-in because these are production-width CUDA acceptance shapes, not tiny
// unit fixtures. Run in a fresh process with MLX_ENABLE_TF32=0 (also on Metal).
func TestGridStrictDilatedTrainingPrediction(t *testing.T) {
	if os.Getenv("MIXLAB_TEST_WIDE_DILATION") != "1" {
		t.Skip("set MIXLAB_TEST_WIDE_DILATION=1 for wide convolution acceptance")
	}
	if !mlxAvailable() {
		t.Fatal("acceptance requires MLX")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Fatal("acceptance requires numpy")
	}
	for _, h := range []int{32, 64} {
		for _, dilation := range []int{2, 4} {
			t.Run(fmt.Sprintf("h%d_d%d", h, dilation), func(t *testing.T) {
				cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
				if err != nil {
					t.Fatal(err)
				}
				no := false
				cfg.Training.AllowTF32 = &no
				cfg.Training.Steps, cfg.Training.WarmupSteps = 3, 0
				cfg.InputAdapter.Channels, cfg.InputAdapter.Height, cfg.InputAdapter.Width = 192, h, h
				block := &cfg.Blocks[0]
				block.Weights[0].Shape = []string{"192", "3", "3", "192"}
				block.Weights[1].Shape = []string{"192"}
				proj, bias := block.Weights[0], block.Weights[1]
				proj.Name, proj.Shape = "proj.weight", []string{"1", "1", "1", "192"}
				bias.Name, bias.Shape = "proj.bias", []string{"1"}
				bias.Init = nil // Avoid sharing/mutating the paired conv initializer.
				block.Weights = append(block.Weights, proj, bias)
				block.Ops[0].Output = "dilated"
				block.Ops[0].Params["dilation"], block.Ops[0].Params["padding"] = dilation, dilation
				block.Ops = append(block.Ops, arch.OpSpec{Op: "conv2d", Inputs: []string{"dilated", "proj.weight", "proj.bias"}, Output: "prediction", Params: map[string]interface{}{"kernel": 1, "stride": 1, "padding": 0}})
				dir := t.TempDir()
				x, y := make([]float32, 2*192*h*h), make([]float32, 2*h*h)
				for i := range x {
					x[i] = float32(math.Sin(float64(i) * .17))
				}
				writeNPYFloat32(t, filepath.Join(dir, "x.npy"), []int{2, 192, h, h}, x)
				writeNPYFloat32(t, filepath.Join(dir, "y.npy"), []int{2, 1, h, h}, y)
				mask := make([]float32, len(y))
				for i := range mask {
					mask[i] = 1
				}
				writeNPYFloat32(t, filepath.Join(dir, "m.npy"), []int{2, 1, h, h}, mask)
				source := filepath.Join(dir, "source.json")
				if err = atomicWriteJSON(source, map[string]any{"splits": map[string]any{"train": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy"}, "val": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy"}}}); err != nil {
					t.Fatal(err)
				}
				prepared := filepath.Join(dir, "prepared")
				if err = RunPrepare(PrepareOptions{Input: source, Output: prepared, InputFormat: "grid"}); err != nil {
					t.Fatal(err)
				}
				config, checkpoint := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.safetensors")
				if err = atomicWriteJSON(config, cfg); err != nil {
					t.Fatal(err)
				}
				manifest := filepath.Join(prepared, data.DatasetManifestFilename)
				if err = RunArch(config, manifest, TrainOptions{SafetensorsPath: checkpoint, LogEvery: 1, ValEvery: 1}); err != nil {
					t.Fatal(err)
				}
				if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: checkpoint, Input: manifest, Split: "val", Output: filepath.Join(dir, "predictions")}); err != nil {
					t.Fatal(err)
				}
				out, err := exec.Command("python3", "-c", `
import pathlib, sys, numpy as np
files = list(pathlib.Path(sys.argv[1]).glob("*.npy"))
assert len(files) == 2
for path in files:
    values = np.load(path)
    assert values.shape == (1, int(sys.argv[2]), int(sys.argv[2]))
    assert np.isfinite(values).all()
`, filepath.Join(dir, "predictions"), strconv.Itoa(h)).CombinedOutput()
				if err != nil {
					t.Fatalf("prediction validation: %v: %s", err, out)
				}
			})
		}
	}
}
