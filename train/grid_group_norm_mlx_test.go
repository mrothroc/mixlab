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

func TestGridGroupNormTrainingAndTorchExport(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	cfg, err := LoadArchConfig("../examples/grid_group_norm_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 42, "", 0)
	for i, v := range weights[2] {
		if v != 1 || weights[3][i] != 0 {
			t.Fatal("incorrect affine initialization")
		}
	}
	prog, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	tr, err := initGPUTrainer(prog, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	b := data.GridBatch{BatchSize: 1, Count: 1, Inputs: make([]float32, 8*8*2), Targets: make([]float32, 8*8), LossMask: make([]float32, 8*8)}
	for i := range b.Inputs {
		b.Inputs[i] = float32(math.Sin(float64(i) * .13))
	}
	for i := range b.Targets {
		b.Targets[i], b.LossMask[i] = .3*b.Inputs[2*i]+.1, 1
	}
	var first, last float32
	for step := 0; step < 100; step++ {
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 1, 0, .001); err != nil {
			t.Fatal(err)
		}
		last, err = tr.CollectLossGPU()
		if err != nil || math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
			t.Fatalf("step %d: loss=%g err=%v", step, last, err)
		}
		if step == 0 {
			first = last
		}
	}
	if last >= first {
		t.Fatalf("loss did not decrease: %g -> %g", first, last)
	}
	t.Logf("batch=1 GroupNorm training loss %g -> %g", first, last)
	if _, err = tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 1, 0, []string{"predictions"}); err != nil {
		t.Fatal(err)
	}
	native, err := readTrainerOutput(tr, "predictions", []int{1, 8, 8, 1})
	if err != nil {
		t.Fatal(err)
	}
	live, err := readTrainerWeights(tr)
	if err != nil {
		t.Fatal(err)
	}
	for _, index := range []int{0, 2, 3, 4} {
		changed := false
		for j, v := range live[index] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("non-finite weight")
			}
			changed = changed || v != weights[index][j]
		}
		if !changed {
			t.Fatalf("weight %s did not train", shapes[index].Name)
		}
	}
	t.Run("TorchExport", func(t *testing.T) {
		if err := exec.Command("python3", "-c", "import torch, safetensors").Run(); err != nil {
			t.Skip("PyTorch and safetensors required")
		}
		dir := t.TempDir()
		config, checkpoint, mapping := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.st"), filepath.Join(dir, "map.json")
		if err = atomicWriteJSON(config, cfg); err != nil {
			t.Fatal(err)
		}
		if err = exportSafetensors(checkpoint, cfg, shapes, live); err != nil {
			t.Fatal(err)
		}
		m := torchStateMap{Format: "mixlab.torch_state_map.v1"}
		for i, name := range []string{"conv.weight", "conv.bias", "norm.weight", "norm.bias", "out.weight", "out.bias"} {
			entry := torchStateMapping{Source: shapes[i].Name, Target: name, Transform: "identity", Shape: shapes[i].Shape}
			if len(entry.Shape) == 4 {
				s := entry.Shape
				entry.Transform, entry.Axes, entry.Shape = "transpose", []int{0, 3, 1, 2}, []int{s[0], s[3], s[1], s[2]}
			}
			m.Mappings = append(m.Mappings, entry)
		}
		if err = atomicWriteJSON(mapping, m); err != nil {
			t.Fatal(err)
		}
		pkg := filepath.Join(dir, "export")
		if err = RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: mapping, OutputDir: pkg}); err != nil {
			t.Fatal(err)
		}
		ref := filepath.Join(dir, "native.json")
		if err = atomicWriteJSON(ref, map[string]any{"inputs": b.Inputs, "predictions": native}); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command("python3", "../scripts/check_grid_group_norm_export.py", pkg, ref).CombinedOutput()
		if err != nil {
			t.Fatalf("export parity: %v: %s", err, out)
		}
		t.Log(string(out))
	})
}
