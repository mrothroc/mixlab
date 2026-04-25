//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestOptimizerOverride_FrozenWeight(t *testing.T) {
	fixture := newOptimizerOverrideFixture(t)

	finalPath := filepath.Join(fixture.dir, "frozen_final.safetensors")
	err := RunArch(fixture.configPath, fixture.trainPattern, TrainOptions{
		SafetensorsLoad: fixture.initialWeightsPath,
		SafetensorsPath: finalPath,
		OptimizerOverride: func(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
			if len(spec.Weights) == 0 {
				t.Fatal("default optimizer spec had no weights")
			}
			frozenGroup := spec.Groups[spec.Weights[0].GroupIndex]
			frozenGroup.LR = 0
			spec.Groups = append(spec.Groups, frozenGroup)
			spec.Weights[0].GroupIndex = len(spec.Groups) - 1
			return spec, nil
		},
	})
	if err != nil {
		t.Fatalf("RunArch: %v", err)
	}

	initialWeights, err := loadSafetensorsWeights(fixture.initialWeightsPath, fixture.shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights(initial): %v", err)
	}
	finalWeights, err := loadSafetensorsWeights(finalPath, fixture.shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights(final): %v", err)
	}

	if !equalFloat32Slices(initialWeights[0], finalWeights[0]) {
		t.Fatalf("frozen weight %q changed despite LR=0 override", fixture.shapes[0].Name)
	}

	changed := false
	for i := 1; i < len(initialWeights); i++ {
		if !equalFloat32Slices(initialWeights[i], finalWeights[i]) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("all non-frozen weights remained unchanged; expected at least one trainable weight to update")
	}
}

func TestOptimizerOverride_IncompleteCoverage(t *testing.T) {
	fixture := newOptimizerOverrideFixture(t)

	err := RunArch(fixture.configPath, fixture.trainPattern, TrainOptions{
		SafetensorsLoad: fixture.initialWeightsPath,
		OptimizerOverride: func(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) (gpu.TrainerOptimizerSpec, error) {
			spec.Weights = spec.Weights[:len(spec.Weights)-1]
			return spec, nil
		},
	})
	if err == nil {
		t.Fatal("RunArch succeeded with incomplete optimizer coverage, want error")
	}
	if !strings.Contains(err.Error(), "missing optimizer assignments") {
		t.Fatalf("error %q did not mention missing optimizer assignments", err)
	}
	if !strings.Contains(err.Error(), fixture.shapes[len(fixture.shapes)-1].Name) {
		t.Fatalf("error %q did not mention missing weight %q", err, fixture.shapes[len(fixture.shapes)-1].Name)
	}
}

func TestOptimizerOverride_NilDefaultBehavior(t *testing.T) {
	fixture := newOptimizerOverrideFixture(t)

	baselinePath := filepath.Join(fixture.dir, "baseline_final.safetensors")
	baselineResult, err := runTrain(fixture.cfg, fixture.trainPattern, TrainOptions{
		SafetensorsLoad: fixture.initialWeightsPath,
		SafetensorsPath: baselinePath,
	})
	if err != nil {
		t.Fatalf("runTrain baseline: %v", err)
	}

	overridePath := filepath.Join(fixture.dir, "override_nil_final.safetensors")
	overrideResult, err := runTrain(fixture.cfg, fixture.trainPattern, TrainOptions{
		SafetensorsLoad:   fixture.initialWeightsPath,
		SafetensorsPath:   overridePath,
		OptimizerOverride: nil,
	})
	if err != nil {
		t.Fatalf("runTrain nil override: %v", err)
	}

	if diff := math.Abs(baselineResult.FirstLoss - overrideResult.FirstLoss); diff > 1e-7 {
		t.Fatalf("first loss mismatch: baseline=%.9f override=%.9f diff=%.9g", baselineResult.FirstLoss, overrideResult.FirstLoss, diff)
	}
	if diff := math.Abs(baselineResult.LastLoss - overrideResult.LastLoss); diff > 1e-7 {
		t.Fatalf("last loss mismatch: baseline=%.9f override=%.9f diff=%.9g", baselineResult.LastLoss, overrideResult.LastLoss, diff)
	}

	baselineWeights, err := loadSafetensorsWeights(baselinePath, fixture.shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights(baseline): %v", err)
	}
	overrideWeights, err := loadSafetensorsWeights(overridePath, fixture.shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights(override): %v", err)
	}
	for i := range baselineWeights {
		if !equalFloat32Slices(baselineWeights[i], overrideWeights[i]) {
			t.Fatalf("weight %q differed between baseline and nil override runs", fixture.shapes[i].Name)
		}
	}
}

type optimizerOverrideFixture struct {
	dir                string
	cfg                *ArchConfig
	configPath         string
	trainPattern       string
	initialWeightsPath string
	shapes             []WeightShape
}

func newOptimizerOverrideFixture(t *testing.T) optimizerOverrideFixture {
	t.Helper()

	dir := t.TempDir()
	cfg := &ArchConfig{
		Name:      "optimizer_override_test",
		ModelDim:  16,
		VocabSize: 32,
		SeqLen:    4,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 2},
			{Type: "swiglu"},
		},
		Training: DefaultTrainingSpec(),
	}
	cfg.Training.Steps = 2
	cfg.Training.LR = 1e-3
	cfg.Training.Seed = 7
	cfg.Training.BatchTokens = 8

	configPath := filepath.Join(dir, "config.json")
	configBlob, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		t.Fatalf("Marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, configBlob, 0o644); err != nil {
		t.Fatalf("Write config: %v", err)
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	initialWeights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	initialWeightsPath := filepath.Join(dir, "initial.safetensors")
	if err := exportSafetensors(initialWeightsPath, cfg, shapes, initialWeights); err != nil {
		t.Fatalf("exportSafetensors(initial): %v", err)
	}

	trainDir := filepath.Join(dir, "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", trainDir, err)
	}
	tokens := []uint16{1, 2, 3, 4, 5, 6, 7, 8, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), tokens)

	return optimizerOverrideFixture{
		dir:                dir,
		cfg:                cfg,
		configPath:         configPath,
		trainPattern:       filepath.Join(trainDir, "train_*.bin"),
		initialWeightsPath: initialWeightsPath,
		shapes:             shapes,
	}
}

func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
