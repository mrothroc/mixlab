//go:build mlx && cgo && (darwin || linux)

package train

import (
	"bytes"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGridVectorsTrainingResumePredictionExport(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGridVectors(t)
	cfg, err := LoadArchConfig("../examples/grid_vectors_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Steps = 30
	cfg.Training.WarmupSteps = 0
	cfg.Training.HoldSteps = 0
	dir := t.TempDir()
	config, checkpoint := filepath.Join(dir, "config.json"), filepath.Join(dir, "weights.st")
	if err = atomicWriteJSON(config, cfg); err != nil {
		t.Fatal(err)
	}
	fullDir := filepath.Join(dir, "full")
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	initial := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	var initialMSE float64
	for row := 0; row < 3; row++ {
		prediction := initial[0][0]*(float32(2*(row+1))+.125) + initial[1][0]
		difference := float64(prediction - (float32(row+1) + .25))
		initialMSE += difference * difference / 3
	}
	full, err := runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: fullDir, CheckpointEvery: 7, ValEvery: 7, SafetensorsPath: checkpoint, LogEvery: 15})
	if err != nil {
		t.Fatal(err)
	}
	// Compare the same complete validation split, not differently shuffled
	// first/last batches (the final batch has only one real record).
	if !full.HasValLoss || !isFiniteLoss(full.LastLoss) || !isFiniteLoss(full.LastValLoss) || full.LastValLoss >= initialMSE {
		t.Fatalf("bad training result: %+v", full)
	}
	t.Logf("vector validation MSE %g -> %g", initialMSE, full.LastValLoss)
	weights, err := loadSafetensorsWeights(checkpoint, shapes)
	if err != nil {
		t.Fatal(err)
	}
	resumeDir := filepath.Join(dir, "resume")
	resumeFinal := filepath.Join(dir, "resumed.st")
	resumed, err := runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(7)), CheckpointDir: resumeDir, CheckpointEvery: 7, ValEvery: 7, SafetensorsPath: resumeFinal})
	if err != nil {
		t.Fatal(err)
	}
	if resumed.LastLoss != full.LastLoss || resumed.LastValLoss != full.LastValLoss {
		t.Fatal("resume loss changed", full, resumed)
	}
	a, err := resolveResumeManifest(fullDir)
	if err != nil {
		t.Fatal(err)
	}
	b, err := resolveResumeManifest(resumeDir)
	if err != nil {
		t.Fatal(err)
	}
	sa, err := loadResumeState(a)
	if err != nil {
		t.Fatal(err)
	}
	sb, err := loadResumeState(b)
	if err != nil {
		t.Fatal(err)
	}
	resumedWeights, err := loadSafetensorsWeights(resumeFinal, shapes)
	if err != nil || !reflect.DeepEqual(weights, resumedWeights) {
		t.Fatal("vector replay final weights mismatch", err)
	}
	if !reflect.DeepEqual(sa.Trainer, sb.Trainer) || !reflect.DeepEqual(a.Grid, b.Grid) {
		t.Fatal("vector replay optimizer/sampler state mismatch")
	}
	if err = runEvalModeWithOptions(config, manifest, checkpoint, EvalModeOptions{}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "predictions")
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: checkpoint, Input: manifest, Output: out}); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(out, "*.npy"))
	if err != nil || len(files) != 3 {
		t.Fatal(files, err)
	}
	var predictions []float32
	for row, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		y, err := decodeNPY(raw)
		if err != nil || !reflect.DeepEqual(y.Shape, []int{1, 1, 1}) {
			t.Fatal(y.Shape, err)
		}
		want := weights[0][0]*(float32(2*(row+1))+.125) + weights[1][0]
		if math.Abs(float64(y.F32[0]-want)) > 1e-5 {
			t.Fatalf("vector-dependent CPU oracle: %g != %g", y.F32[0], want)
		}
		predictions = append(predictions, y.F32[0])
	}
	inputSource := filepath.Join(dir, "input-only.json")
	sourceDir := filepath.Dir(filepath.Dir(manifest))
	if err = atomicWriteJSON(inputSource, map[string]any{"splits": map[string]any{"predict": map[string]string{"inputs": filepath.Join(sourceDir, "x.npy"), "vectors": filepath.Join(sourceDir, "v.npy")}}}); err != nil {
		t.Fatal(err)
	}
	prepared := filepath.Join(dir, "input-only")
	if err = runPrepare(PrepareOptions{Input: inputSource, Output: prepared, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: checkpoint, Input: filepath.Join(prepared, "mixlab.dataset.json"), Output: filepath.Join(dir, "input-only-predictions")}); err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		labelled, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		unlabelled, err := os.ReadFile(filepath.Join(dir, "input-only-predictions", filepath.Base(file)))
		if err != nil || !bytes.Equal(labelled, unlabelled) {
			t.Fatal("prediction changed when targets were omitted", err)
		}
	}
	t.Run("TorchExport", func(t *testing.T) {
		if err := exec.Command("python3", "-c", "import torch, safetensors").Run(); err != nil {
			t.Skip("PyTorch and safetensors required")
		}
		mapping := filepath.Join(dir, "map.json")
		m := torchStateMap{Format: "mixlab.torch_state_map.v1", Mappings: []torchStateMapping{
			{Source: shapes[0].Name, Target: "conv.weight", Transform: "transpose", Axes: []int{0, 3, 1, 2}, Shape: []int{1, 1, 1, 1}},
			{Source: shapes[1].Name, Target: "conv.bias", Transform: "identity", Shape: []int{1}},
		}}
		if err := atomicWriteJSON(mapping, m); err != nil {
			t.Fatal(err)
		}
		pkg := filepath.Join(dir, "export")
		if err := RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: mapping, OutputDir: pkg}); err != nil {
			t.Fatal(err)
		}
		ref := filepath.Join(dir, "native.json")
		if err := atomicWriteJSON(ref, map[string]any{"inputs": []float32{1, 2, 3}, "vectors": []float32{1.125, 2.125, 3.125}, "predictions": predictions}); err != nil {
			t.Fatal(err)
		}
		output, err := exec.Command("python3", "../scripts/check_grid_vectors_export.py", pkg, ref).CombinedOutput()
		if err != nil {
			t.Fatalf("export roundtrip: %v: %s", err, output)
		}
		t.Log(string(output))
	})
	// Warm starts share the same vector input contract, without loader state.
	cfg.Training.InitFrom = checkpoint
	cfg.Training.Steps = 2
	if _, err = runGridTrain(cfg, manifest, TrainOptions{}); err != nil {
		t.Fatal(err)
	}
	cfg.Training.InitFrom = ""
	cfg.Training.Steps = 30
	path := filepath.Join(filepath.Dir(manifest), "train_00000.grid")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 1
	if err = os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = runGridTrain(cfg, manifest, TrainOptions{Resume: filepath.Join(fullDir, resumeManifestFilename(7)), ValEvery: 7}); err == nil || !strings.Contains(err.Error(), "dataset") {
		t.Fatal("accepted changed vectors on resume", err)
	}
}

func isFiniteLoss(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
