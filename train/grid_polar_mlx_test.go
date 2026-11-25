//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/internal/gridpoints"
)

func TestGridPolarTrainingResumePredictionExport(t *testing.T) {
	testGridPolarTrainingResumePredictionExport(t, false)
}

func TestGridPointsPolarTrainingResumePredictionExport(t *testing.T) {
	testGridPolarTrainingResumePredictionExport(t, true)
}

func testGridPolarTrainingResumePredictionExport(t *testing.T, points bool) {
	if !runInStrictFP32Process(t) {
		return
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy required")
	}
	cfg, err := LoadArchConfig("../examples/grid_polar_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	if points {
		cfg.InputAdapter.VectorPoints = gridpoints.Columns{{0, 1}}
		cfg.Training.GridAugmentation = &arch.GridAugmentationSpec{Dihedral: true}
	}
	cfg.Training.Steps = 30
	cfg.Training.WarmupSteps = 0
	cfg.Training.HoldSteps = 0
	dir := t.TempDir()
	x := make([]float32, 3*2*64)
	y := make([]float32, 3*64)
	m := make([]float32, len(y))
	nhwc := make([]float32, len(x))
	vectors := []float32{0, 0, 3.25, 4.5, 7, 7}
	for n := 0; n < 3; n++ {
		for i := 0; i < 64; i++ {
			v := float32(math.Sin(float64(i)*.13) + float64(n)*.1)
			x[n*128+i], x[n*128+64+i] = v, 1-v
			nhwc[n*128+2*i], nhwc[n*128+2*i+1] = v, 1-v
			y[n*64+i], m[n*64+i] = .3*v+.2, 1
		}
	}
	for name, values := range map[string][]float32{"x": x, "y": y, "m": m, "v": vectors} {
		shape := []int{3, 1, 8, 8}
		if name == "x" {
			shape[1] = 2
		}
		if name == "v" {
			shape = []int{3, 2}
		}
		writeNPYFloat32(t, filepath.Join(dir, name+".npy"), shape, values)
	}
	entry := map[string]string{"inputs": "x.npy", "targets": "y.npy", "masks": "m.npy", "vectors": "v.npy"}
	source := filepath.Join(dir, "source.json")
	prepared := filepath.Join(dir, "prepared")
	if err = atomicWriteJSON(source, map[string]any{"splits": map[string]any{"train": entry, "val": entry, "predict": entry}}); err != nil {
		t.Fatal(err)
	}
	if err = runPrepare(PrepareOptions{Input: source, Output: prepared, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(prepared, data.DatasetManifestFilename)
	config, checkpoint := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.st")
	if err = atomicWriteJSON(config, cfg); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	initial := initWeightData(shapes, 42, "", 0)
	initialPath := filepath.Join(dir, "initial.st")
	if err = exportSafetensors(initialPath, cfg, shapes, initial); err != nil {
		t.Fatal(err)
	}
	ds, err := data.OpenGridDataset(manifest, "val")
	if err != nil {
		t.Fatal(err)
	}
	defer ds.Close()
	initialMetrics := gridMetrics{}
	if err = visitGridPredictions(cfg, initialPath, ds, initialMetrics.add); err != nil {
		t.Fatal(err)
	}
	fullDir := filepath.Join(dir, "full")
	result, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: fullDir, CheckpointEvery: 7, ValEvery: 7, SafetensorsPath: checkpoint, LogEvery: 15})
	if err != nil {
		t.Fatal(err)
	}
	if !isFiniteLoss(result.LastLoss) || !result.HasValLoss {
		t.Fatal(result)
	}
	var native []float32
	finalMetrics := gridMetrics{}
	if err = visitGridPredictions(cfg, checkpoint, ds, func(b data.GridBatch, p []float32) error {
		native = append(native, p[:b.Count*64]...)
		return finalMetrics.add(b, p)
	}); err != nil {
		t.Fatal(err)
	}
	if finalMetrics.SSE >= initialMetrics.SSE {
		t.Fatalf("loss did not decrease: %+v -> %+v", initialMetrics, finalMetrics)
	}
	t.Logf("fixed-split squared error %+v -> %+v", initialMetrics, finalMetrics)
	weights, err := loadSafetensorsWeights(checkpoint, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < 4; n++ {
		if reflect.DeepEqual(weights[n], initial[n]) {
			t.Fatalf("polar branch weight %d did not train", n)
		}
	}
	resumedPath := filepath.Join(dir, "resumed.st")
	resumeDir := filepath.Join(dir, "resume")
	if points {
		zero := 0
		cfg.Training.GridLoader = &arch.GridLoaderSpec{PrefetchBatches: &zero, ReadWorkers: 1}
	}
	_, err = runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(7)), CheckpointDir: resumeDir, CheckpointEvery: 7, ValEvery: 7, SafetensorsPath: resumedPath})
	if err != nil {
		t.Fatal(err)
	}
	replay, err := loadSafetensorsWeights(resumedPath, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for n := range weights {
		for j, v := range weights[n] {
			if !isFiniteLoss(float64(v)) || !isFiniteLoss(float64(replay[n][j])) || math.Abs(float64(v-replay[n][j])) > 1e-5 {
				t.Fatalf("resume weight %d/%d: %g != %g", n, j, v, replay[n][j])
			}
		}
	}
	a, err := resolveResumeManifest(fullDir)
	if err != nil {
		t.Fatal(err)
	}
	b, err := resolveResumeManifest(resumeDir)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Grid, b.Grid) {
		t.Fatal("sampler replay changed")
	}
	sa, err := loadResumeState(a)
	if err != nil {
		t.Fatal(err)
	}
	sb, err := loadResumeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(sa.Trainer.Optimizer, sb.Trainer.Optimizer) || len(sa.Trainer.Tensors) != len(sb.Trainer.Tensors) {
		t.Fatal("optimizer replay metadata changed")
	}
	for i, x := range sa.Trainer.Tensors {
		y := sb.Trainer.Tensors[i]
		if x.Kind != y.Kind || x.WeightIndex != y.WeightIndex || !reflect.DeepEqual(x.Shape, y.Shape) || len(x.Data) != len(y.Data) {
			t.Fatal("optimizer state layout changed")
		}
		for j, v := range x.Data {
			if !isFiniteLoss(float64(v)) || !isFiniteLoss(float64(y.Data[j])) || math.Abs(float64(v-y.Data[j])) > 1e-5 {
				t.Fatalf("optimizer state mismatch %d/%d", i, j)
			}
		}
	}
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: checkpoint, Input: manifest, Output: filepath.Join(dir, "predictions")}); err != nil {
		t.Fatal(err)
	}
	// Coordinate validation is shared by native training and eval/prediction.
	t.Run("TorchExport", func(t *testing.T) {
		if err := exec.Command("python3", "-c", "import torch,safetensors").Run(); err != nil {
			t.Skip("torch/safetensors required")
		}
		mapping := torchStateMap{Format: "mixlab.torch_state_map.v1"}
		for _, s := range shapes {
			target := strings.TrimPrefix(s.Name, "network.")
			shape := append([]int(nil), s.Shape...)
			transform := "identity"
			var axes []int
			if len(shape) == 4 {
				transform = "transpose"
				axes = []int{0, 3, 1, 2}
				shape = []int{shape[0], shape[3], shape[1], shape[2]}
			}
			mapping.Mappings = append(mapping.Mappings, torchStateMapping{Source: s.Name, Target: target, Shape: shape, Transform: transform, Axes: axes})
		}
		mapPath := filepath.Join(dir, "map.json")
		if err := atomicWriteJSON(mapPath, mapping); err != nil {
			t.Fatal(err)
		}
		pkg := filepath.Join(dir, "export")
		if err := RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: mapPath, OutputDir: pkg}); err != nil {
			t.Fatal(err)
		}
		if points {
			for _, name := range []string{"config.json", "export.json"} {
				raw, err := os.ReadFile(filepath.Join(pkg, name))
				if err != nil {
					t.Fatal(err)
				}
				var m map[string]json.RawMessage
				if err = json.Unmarshal(raw, &m); err != nil {
					t.Fatal(err)
				}
				if name == "config.json" {
					if err = json.Unmarshal(m["input_adapter"], &m); err != nil {
						t.Fatal(err)
					}
				}
				var declared gridpoints.Columns
				if err = json.Unmarshal(m["vector_points"], &declared); err != nil || !reflect.DeepEqual(declared, cfg.InputAdapter.VectorPoints) {
					t.Fatal(name, declared, err)
				}
			}
		}
		ref := filepath.Join(dir, "native.json")
		if err := atomicWriteJSON(ref, map[string]any{"inputs": nhwc, "vectors": vectors, "predictions": native}); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command("python3", "../scripts/check_grid_polar_export.py", pkg, ref).CombinedOutput()
		if err != nil {
			t.Fatalf("%v: %s", err, out)
		}
		t.Log(string(out))
	})
}
