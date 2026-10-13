//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

func TestGridASPPTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy required for grid prepare")
	}
	cfg, err := LoadArchConfig("../examples/grid_aspp_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	const records, pixels = 4, 32 * 32
	x, y := make([]float32, records*2*pixels), make([]float32, records*pixels)
	mask := make([]float32, len(y))
	for i := range mask {
		mask[i] = 1
	}
	for row := 0; row < records; row++ {
		for j := 0; j < pixels; j++ {
			v := float32((row*17+j)%31) / 31
			x[row*2*pixels+j], x[row*2*pixels+pixels+j] = v, 1-v
			y[row*pixels+j] = .4*v + .2
		}
	}
	writeNPYFloat32(t, filepath.Join(dir, "x.npy"), []int{records, 2, 32, 32}, x)
	writeNPYFloat32(t, filepath.Join(dir, "y.npy"), []int{records, 1, 32, 32}, y)
	writeNPYFloat32(t, filepath.Join(dir, "mask.npy"), []int{records, 1, 32, 32}, mask)
	source := filepath.Join(dir, "source.json")
	if err = atomicWriteJSON(source, map[string]any{"splits": map[string]any{"train": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "mask.npy"}, "val": map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "mask.npy"}}}); err != nil {
		t.Fatal(err)
	}
	prepared := filepath.Join(dir, "prepared")
	if err = RunPrepare(PrepareOptions{Input: source, Output: prepared, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(prepared, data.DatasetManifestFilename)
	cfg.Training.WarmupSteps, cfg.Training.HoldSteps = 0, 0
	var timed time.Duration
	var steps int
	r, err := runGridTrainWithStep(cfg, manifest, TrainOptions{ValEvery: 50, LogEvery: 50}, func(tr GPUTrainer, b objectiveBatch, size int, lr float32) (float32, error) {
		start := time.Now()
		if e := submitPreparedStepGPU(tr, b, size, 0, lr); e != nil {
			return 0, e
		}
		loss, e := tr.CollectLossGPU()
		if steps >= 5 {
			timed += time.Since(start)
		}
		steps++
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("non-finite ASPP loss step %d: %g", steps, loss)
		}
		return loss, e
	})
	if err != nil || !r.HasValLoss || r.LastLoss >= r.FirstLoss {
		t.Fatalf("ASPP training: %+v, %v", r, err)
	}
	t.Logf("device=%s batch=2 geometry=32x32 channels=2 precision=float32 steps=%d warmup=5 records/s=%.2f peak_MiB=%.2f loss=%g->%g", gpu.DeviceName(), steps, float64((steps-5)*2)/timed.Seconds(), float64(gpu.MemoryStatsSnapshot().PeakBytes)/(1<<20), r.FirstLoss, r.LastLoss)
	// Exercise the same public entry point as -mode arch, including checkpoints.
	cfg.Training.Steps = 2
	config := filepath.Join(dir, "model.json")
	if err = atomicWriteJSON(config, cfg); err != nil {
		t.Fatal(err)
	}
	if err = RunArch(config, manifest, TrainOptions{SafetensorsPath: filepath.Join(dir, "weights.st"), LogEvery: 1, ValEvery: 1}); err != nil {
		t.Fatal(err)
	}
}

func TestGridDilationTorchExportParity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	if err := exec.Command("python3", "-c", "import torch, safetensors").Run(); err != nil {
		t.Skip("PyTorch and safetensors required for export parity")
	}
	for _, d := range []int{2, 4} {
		t.Run(fmt.Sprint(d), func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Blocks[0].Ops[0].Params["dilation"] = d
			cfg.Blocks[0].Ops[0].Params["padding"] = d
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			weights := initWeightData(shapes, 42, "", 0)
			prog, err := arch.BuildGridIRProgram(cfg, true)
			if err != nil {
				t.Fatal(err)
			}
			tr, err := initGPUTrainer(prog, cfg, weights, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer tr.CloseTrainer()
			b := data.GridBatch{BatchSize: 2, Count: 2, Inputs: make([]float32, 2*8*8*2), Targets: make([]float32, 2*8*8), LossMask: make([]float32, 2*8*8)}
			for i := range b.Inputs {
				b.Inputs[i] = float32(math.Sin(float64(i)))
			}
			for i := range b.LossMask {
				b.LossMask[i] = 1
			}
			if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 2, 0, .001); err != nil {
				t.Fatal(err)
			}
			if _, err = tr.CollectLossGPU(); err != nil {
				t.Fatal(err)
			}
			if _, err = tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 2, 0, []string{"predictions"}); err != nil {
				t.Fatal(err)
			}
			native, err := readTrainerOutput(tr, "predictions", []int{2, 8, 8, 1})
			if err != nil {
				t.Fatal(err)
			}
			weights, err = readTrainerWeights(tr)
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			config, checkpoint, mapping := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.st"), filepath.Join(dir, "map.json")
			if err = atomicWriteJSON(config, cfg); err != nil {
				t.Fatal(err)
			}
			if err = exportSafetensors(checkpoint, cfg, shapes, weights); err != nil {
				t.Fatal(err)
			}
			m := torchStateMap{Format: "mixlab.torch_state_map.v1", Mappings: []torchStateMapping{
				{Source: shapes[0].Name, Target: "weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{1, 2, 3, 3}},
				{Source: shapes[1].Name, Target: "bias", Transform: "identity", Shape: []int{1}},
			}}
			if err = atomicWriteJSON(mapping, m); err != nil {
				t.Fatal(err)
			}
			pkg := filepath.Join(dir, "export")
			if err = RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: mapping, OutputDir: pkg}); err != nil {
				t.Fatal(err)
			}
			if err = atomicWriteJSON(filepath.Join(dir, "native.json"), map[string]any{"inputs": b.Inputs, "predictions": native}); err != nil {
				t.Fatal(err)
			}
			out, err := exec.Command("python3", "../scripts/check_grid_dilation_export.py", pkg, filepath.Join(dir, "native.json")).CombinedOutput()
			if err != nil {
				t.Fatalf("export parity: %v: %s", err, out)
			}
			t.Log(string(out))
		})
	}
}
