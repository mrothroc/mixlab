//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridInitialValidationStopsBeforeUpdate(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	for _, reason := range []string{"target", "val_gt"} {
		t.Run(reason, func(t *testing.T) {
			cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Steps = 2
			cfg.Training.WarmupSteps = 0
			if reason == "target" {
				cfg.Training.TargetValLoss = 100
			} else {
				cfg.Training.EarlyStop = &EarlyStopSpec{ValGT: 1e-10}
			}
			dir := t.TempDir()
			final := filepath.Join(dir, "final.safetensors")
			r, err := runGridTrain(cfg, manifest, TrainOptions{SafetensorsPath: final, CheckpointDir: dir})
			if err != nil {
				t.Fatal(err)
			}
			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			initial, err := loadSafetensorsWeights(filepath.Join(dir, "best.safetensors"), shapes)
			if err != nil {
				t.Fatal(err)
			}
			got, err := loadSafetensorsWeights(final, shapes)
			if err != nil {
				t.Fatal(err)
			}
			if !r.HasValLoss || !reflect.DeepEqual(initial, got) {
				t.Fatal("initial validation stop did not preserve weights")
			}
		})
	}
}

func TestGridEmptyValidationFailsBeforeCheckpoint(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGridWithMask(t, 0)
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "checkpoint")
	_, err = runGridTrain(cfg, manifest, TrainOptions{CheckpointDir: dir})
	if err == nil || !strings.Contains(err.Error(), "zero valid") {
		t.Fatalf("error=%v", err)
	}
	if _, err = os.Stat(filepath.Join(dir, "best.safetensors")); !os.IsNotExist(err) {
		t.Fatal("empty validation saved a best checkpoint")
	}
}

func TestGridAdamEmptyMaskAndFrozenWeights(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	for _, mask := range []float32{0, 1} {
		cfg, err := arch.ParseArchConfig([]byte(`{"name":"grid_adam","input_adapter":{"kind":"grid","channels":1,"height":1,"width":1},"dense_regression":{"output":"net.y","target_channels":1},"blocks":[{"type":"custom","name":"net","weights":[{"name":"w","shape":["1","1","1","1"]},{"name":"bias","shape":["1"]},{"name":"unused","shape":["1"]}],"ops":[{"op":"conv2d","inputs":["x","w","bias"],"output":"y","params":{"kernel":1}}]}],"training":{"objective":"dense_regression","batch_size":1,"optimizer":"adamw","lr":0.1,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weight_decay":0}}`), "grid_adam")
		if err != nil {
			t.Fatal(err)
		}
		p, err := arch.BuildGridIRProgram(cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		tr, err := initGPUTrainer(p, cfg, [][]float32{{2}, {0}, {9}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		b := data.GridBatch{Inputs: []float32{1}, Targets: []float32{0}, LossMask: []float32{mask}, BatchSize: 1, Count: 1}
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 1, 0, .1); err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		loss, err := tr.CollectLossGPU()
		if err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		w, err := readTrainerWeights(tr)
		if err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		state, err := tr.(gpuTrainerStateReader).ReadTrainerState()
		if err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		tr.CloseTrainer()
		if loss != 4*mask || math.Abs(float64(w[0][0]-(2-.1*mask))) > 1e-6 || math.Abs(float64(w[1][0]+.1*mask)) > 1e-6 || w[2][0] != 9 {
			t.Fatalf("mask=%g loss=%g weights=%v", mask, loss, w)
		}
		for _, s := range state.Tensors {
			if s.WeightIndex == 2 {
				t.Fatal("frozen weight has optimizer state")
			}
		}
	}
}

func TestGridTrainingValidationPrediction(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Steps = 30
	cfg.Training.LR = .01
	cfg.Training.WarmupSteps = 0
	cfg.Training.HoldSteps = 0
	dir := t.TempDir()
	config := filepath.Join(dir, "model.json")
	raw, err := os.ReadFile("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(config, raw, 0600); err != nil {
		t.Fatal(err)
	}
	weights := filepath.Join(dir, "weights.safetensors")
	r, err := runGridTrain(cfg, manifest, TrainOptions{SafetensorsPath: weights, CheckpointDir: filepath.Join(dir, "ckpt"), LogEvery: 10, ValEvery: 10})
	if err != nil {
		t.Fatal(err)
	}
	if !r.HasValLoss || r.LastLoss >= r.FirstLoss {
		t.Fatalf("loss did not decrease: %+v", r)
	}
	if err = runEvalModeWithOptions(config, manifest, weights, EvalModeOptions{}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "predictions")
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: weights, Input: manifest, Output: out}); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(out, "*.npy"))
	if err != nil || len(files) != 3 {
		t.Fatalf("files=%v err=%v", files, err)
	}
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: weights, Input: manifest, Output: out}); err == nil {
		t.Fatal("overwrote predictions")
	}
	checkTinyGridPredictions(t, cfg, manifest, weights, files)
	// Inference does not require targets or masks in the input artifact.
	sourceDir := filepath.Dir(filepath.Dir(manifest))
	source, _ := json.Marshal(map[string]any{"splits": map[string]any{"predict": map[string]string{"inputs": filepath.Join(sourceDir, "x.npy")}}})
	input := filepath.Join(dir, "input-only.json")
	if err = os.WriteFile(input, source, 0600); err != nil {
		t.Fatal(err)
	}
	prepared := filepath.Join(dir, "input-only")
	if err = runPrepare(PrepareOptions{Input: input, Output: prepared, InputFormat: "grid"}); err != nil {
		t.Fatal(err)
	}
	if err = RunPredictGrid(PredictGridOptions{ConfigPath: config, SafetensorsLoad: weights, Input: filepath.Join(prepared, data.DatasetManifestFilename), Output: filepath.Join(dir, "input-only-predictions")}); err != nil {
		t.Fatal(err)
	}
	stages, err := filepath.Glob(filepath.Join(dir, ".grid-predict-*"))
	if err != nil || len(stages) != 0 {
		t.Fatal("staging artifacts leaked", stages, err)
	}
}

func checkTinyGridPredictions(t *testing.T, cfg *ArchConfig, manifest, checkpoint string, files []string) {
	t.Helper()
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := loadSafetensorsWeights(checkpoint, shapes)
	if err != nil {
		t.Fatal(err)
	}
	ds, err := data.OpenGridDataset(manifest, "predict")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ds.Close() }()
	for row, file := range files {
		b, err := ds.ReadBatch([]int{row}, 1)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		y, err := decodeNPY(raw)
		if err != nil || len(y.Shape) != 3 || y.Shape[0] != 1 || y.Shape[1] != 8 || y.Shape[2] != 8 {
			t.Fatal("prediction NPY shape", y.Shape, err)
		}
		for h := 0; h < 8; h++ {
			for x := 0; x < 8; x++ {
				want := float64(w[1][0])
				for kh := 0; kh < 3; kh++ {
					for kw := 0; kw < 3; kw++ {
						i, j := h+kh-1, x+kw-1
						if i < 0 || i >= 8 || j < 0 || j >= 8 {
							continue
						}
						for c := 0; c < 2; c++ {
							want += float64(b.Inputs[(i*8+j)*2+c]) * float64(w[0][(kh*3+kw)*2+c])
						}
					}
				}
				if got := float64(y.F32[h*8+x]); math.Abs(got-want) > 1e-5 {
					t.Fatalf("record %d pixel (%d,%d): %g != CPU %g", row, h, x, got, want)
				}
			}
		}
	}
}
