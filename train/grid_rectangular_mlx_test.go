//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridRectangularTrainingTorchExport(t *testing.T) {
	// The example config sets allow_tf32=false, which needs its own process.
	if !runInStrictFP32Process(t) {
		return
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	cfg, err := LoadArchConfig("../examples/grid_rectangular_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 42, "", 0)
	program, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := initGPUTrainer(program, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	b := data.GridBatch{BatchSize: 2, Count: 2, Inputs: make([]float32, 2*8*8*2), Targets: make([]float32, 2*8*8), LossMask: make([]float32, 2*8*8)}
	for i := range b.Targets {
		v := float32(math.Sin(float64(i) * .17))
		b.Inputs[2*i], b.Inputs[2*i+1] = v, 1-v
		b.Targets[i], b.LossMask[i] = .4*v+.2, 1
	}
	eval := func() []float32 {
		t.Helper()
		if _, e := tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 2, 0, []string{"predictions"}); e != nil {
			t.Fatal(e)
		}
		out, e := readTrainerOutput(tr, "predictions", []int{2, 8, 8, 1})
		if e != nil {
			t.Fatal(e)
		}
		return out
	}
	mse := func(predictions []float32) float64 {
		var sum float64
		for i, p := range predictions {
			d := float64(p - b.Targets[i])
			sum += d * d
		}
		return sum / float64(len(predictions))
	}
	before := mse(eval())
	for step := 0; step < 30; step++ {
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 2, 0, .01); err != nil {
			t.Fatal(err)
		}
		loss, e := tr.CollectLossGPU()
		if e != nil || !isFiniteLoss(float64(loss)) {
			t.Fatalf("step %d loss=%g: %v", step, loss, e)
		}
	}
	native := eval()
	if after := mse(native); !isFiniteLoss(after) || after >= before {
		t.Fatalf("fixed-batch MSE did not improve: %g -> %g", before, after)
	} else {
		t.Logf("fixed-batch MSE %g -> %g", before, after)
	}
	weights, err = readTrainerWeights(tr)
	if err != nil {
		t.Fatal(err)
	}
	t.Run("TorchExport", func(t *testing.T) {
		if err := exec.Command("python3", "-c", "import torch, safetensors").Run(); err != nil {
			t.Skip("PyTorch and safetensors required")
		}
		dir := t.TempDir()
		config, checkpoint, mapping := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.st"), filepath.Join(dir, "map.json")
		if err := atomicWriteJSON(config, cfg); err != nil {
			t.Fatal(err)
		}
		if err := exportSafetensors(checkpoint, cfg, shapes, weights); err != nil {
			t.Fatal(err)
		}
		m := torchStateMap{Format: "mixlab.torch_state_map.v1", Mappings: []torchStateMapping{
			{Source: shapes[0].Name, Target: "row.weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{2, 2, 1, 9}},
			{Source: shapes[1].Name, Target: "row.bias", Transform: "identity", Shape: []int{2}},
			{Source: shapes[2].Name, Target: "col.weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{1, 2, 9, 1}},
			{Source: shapes[3].Name, Target: "col.bias", Transform: "identity", Shape: []int{1}},
		}}
		if err := atomicWriteJSON(mapping, m); err != nil {
			t.Fatal(err)
		}
		pkg := filepath.Join(dir, "export")
		if err := RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: mapping, OutputDir: pkg}); err != nil {
			t.Fatal(err)
		}
		ref := filepath.Join(dir, "native.json")
		if err := atomicWriteJSON(ref, map[string]any{"inputs": b.Inputs, "predictions": native}); err != nil {
			t.Fatal(err)
		}
		output, err := exec.Command("python3", "../scripts/check_grid_rectangular_export.py", pkg, ref).CombinedOutput()
		if err != nil {
			t.Fatalf("export roundtrip: %v: %s", err, output)
		}
		t.Log(string(output))
	})
}
