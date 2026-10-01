//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func TestGridFreezeExcludesAutogradAndClipping(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Freeze = []string{"network.conv.weight"}
	cfg.Training.LR = 1
	cfg.Training.ScalarLR = 1
	cfg.Training.Epsilon = 1
	cfg.Training.GradClip = 1
	p, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	w := [][]float32{make([]float32, 18), {0}}
	w[0][8] = 1 // center, first input channel
	tr, err := initGPUTrainer(p, cfg, w, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.CloseTrainer()
	b := data.GridBatch{Inputs: make([]float32, 256), Targets: make([]float32, 128), LossMask: make([]float32, 128), Count: 2, BatchSize: 2}
	for j := range b.Targets {
		b.Inputs[j*2] = 1000
		b.LossMask[j] = 1
	}
	if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 2, 0, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = tr.CollectLossGPU(); err != nil {
		t.Fatal(err)
	}
	got, err := readTrainerWeights(tr)
	if err != nil {
		t.Fatal(err)
	}
	// Only the bias gradient enters the norm: clipped gradient=1, Adam
	// normalized update=1/(1+epsilon)=0.5. Kernel gradients would dominate.
	if !reflect.DeepEqual(got[0], w[0]) || math.Abs(float64(got[1][0]+.5)) > 1e-5 {
		t.Fatalf("frozen gradient affected update: %v", got)
	}
	state, err := tr.(gpuTrainerStateReader).ReadTrainerState()
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range state.Tensors {
		if v.WeightIndex == 0 {
			t.Fatal("frozen kernel has optimizer state")
		}
	}
}

func TestGridAugmentationDisabledParity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	manifest := prepareTinyGrid(t)
	var before [][]float32
	var baseline TrainResult
	for _, explicitFalse := range []bool{false, true} {
		cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
		if err != nil {
			t.Fatal(err)
		}
		cfg.Training.Steps = 3
		if explicitFalse {
			cfg.Training.GridAugmentation = &arch.GridAugmentationSpec{}
		}
		out := filepath.Join(t.TempDir(), "weights.st")
		result, err := runGridTrain(cfg, manifest, TrainOptions{SafetensorsPath: out, ValEvery: 2})
		if err != nil {
			t.Fatal(err)
		}
		shapes, err := computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
		weights, err := loadSafetensorsWeights(out, shapes)
		if err != nil {
			t.Fatal(err)
		}
		if explicitFalse {
			if !reflect.DeepEqual(before, weights) || baseline.FirstLoss != result.FirstLoss || baseline.LastLoss != result.LastLoss || baseline.LastValLoss != result.LastValLoss {
				t.Fatal("disabled augmentation changed training")
			}
		} else {
			before = weights
			baseline = result
		}
	}
}

func TestGridFreezePreservesForward(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	var baseline float32
	for _, freeze := range []bool{false, true} {
		cfg, err := LoadArchConfig("../examples/grid_regression_tiny.json")
		if err != nil {
			t.Fatal(err)
		}
		if freeze {
			cfg.Training.Freeze = []string{"network.conv.weight"}
		}
		p, err := arch.BuildGridIRProgram(cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		shapes, err := computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
		w := initWeightData(shapes, 42, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
		tr, err := initGPUTrainer(p, cfg, w, nil)
		if err != nil {
			t.Fatal(err)
		}
		b := data.GridBatch{Inputs: make([]float32, 256), Targets: make([]float32, 128), LossMask: make([]float32, 128), BatchSize: 2, Count: 2}
		for j := range b.Inputs {
			b.Inputs[j] = float32(j%31) / 31
		}
		for j := range b.LossMask {
			b.LossMask[j] = 1
		}
		eval, err := tr.(gpuObjectiveEvaluator).EvaluateObjectiveGPU(objectiveBatch{grid: &b}, 2, 0)
		if err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 2, 0, 0); err != nil {
			tr.CloseTrainer()
			t.Fatal(err)
		}
		loss, err := tr.CollectLossGPU()
		tr.CloseTrainer()
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(float64(loss-eval)) > 1e-6 {
			t.Fatalf("training/eval compute precision differs: %g %g", loss, eval)
		}
		if freeze && loss != baseline {
			t.Fatalf("freezing changed forward: %g vs %g", loss, baseline)
		}
		baseline = loss
	}
}
