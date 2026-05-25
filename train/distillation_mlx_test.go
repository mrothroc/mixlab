//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestDistillationTeacherRuntimeProducesNormalizedProbs(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	fixture := writeTinyDistillationTeacherFixture(t)
	student := fixture.studentConfig()
	student.Training.Distillation = &DistillationSpec{
		TeacherCheckpoints: []string{fixture.teacherWeightsPath, fixture.teacherWeightsPath},
		TeacherConfigs:     []string{fixture.teacherConfigPath, fixture.teacherConfigPath},
		LossWeightKL:       0.5,
		LossWeightCE:       0.5,
		EnsembleStrategy:   "mean_logits",
	}
	ensemble, err := newDistillationEnsemble(student)
	if err != nil {
		t.Fatalf("newDistillationEnsemble: %v", err)
	}
	defer ensemble.Close()

	batch := objectiveBatch{
		x: []int{1, 2, 3, 4, 4, 3, 2, 1},
		y: []int{2, 3, 4, 5, 3, 2, 1, 0},
	}
	probs, err := ensemble.TeacherProbs(batch, 2, student.SeqLen)
	if err != nil {
		t.Fatalf("TeacherProbs: %v", err)
	}
	if len(probs) != student.Training.BatchTokens*student.VocabSize {
		t.Fatalf("teacher probs len=%d want %d", len(probs), student.Training.BatchTokens*student.VocabSize)
	}
	for row := 0; row < student.Training.BatchTokens; row++ {
		var sum float64
		for _, p := range probs[row*student.VocabSize : (row+1)*student.VocabSize] {
			if p < 0 || math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
				t.Fatalf("row %d invalid probability %g", row, p)
			}
			sum += float64(p)
		}
		if diff := math.Abs(sum - 1); diff > 1e-5 {
			t.Fatalf("row %d probability sum=%g diff=%g", row, sum, diff)
		}
	}
}

func TestDistillationTinyTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	fixture := writeTinyDistillationTeacherFixture(t)
	trainDir := filepath.Join(fixture.dir, "data")
	if err := os.MkdirAll(trainDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", trainDir, err)
	}
	writeInferenceShard(t, filepath.Join(trainDir, "train_000.bin"), []uint16{
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
	})
	trainPattern := filepath.Join(trainDir, "train_*.bin")

	baseline := fixture.studentConfig()
	baselineResult, err := runTrain(baseline, trainPattern, TrainOptions{LogEvery: 0, ValEvery: 0})
	if err != nil {
		t.Fatalf("baseline runTrain: %v", err)
	}

	distilled := fixture.studentConfig()
	distilled.Training.Distillation = &DistillationSpec{
		TeacherCheckpoints: []string{fixture.teacherWeightsPath, fixture.teacherWeightsPath},
		TeacherConfigs:     []string{fixture.teacherConfigPath, fixture.teacherConfigPath},
		LossWeightKL:       0.5,
		LossWeightCE:       0.5,
		EnsembleStrategy:   "mean_logits",
	}
	distilledResult, err := runTrain(distilled, trainPattern, TrainOptions{LogEvery: 0, ValEvery: 0})
	if err != nil {
		t.Fatalf("distilled runTrain: %v", err)
	}
	for name, v := range map[string]float64{
		"baseline last":  baselineResult.LastLoss,
		"distilled last": distilledResult.LastLoss,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s loss is non-finite: %g", name, v)
		}
	}
	if distilledResult.LastLoss > baselineResult.LastLoss+2.0 {
		t.Fatalf("distilled loss=%g unexpectedly far above baseline=%g", distilledResult.LastLoss, baselineResult.LastLoss)
	}
}

type tinyDistillationFixture struct {
	dir                string
	teacherConfigPath  string
	teacherWeightsPath string
}

func (f tinyDistillationFixture) studentConfig() *ArchConfig {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "distill_student",
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2}
		],
		"training": {
			"steps": 2,
			"lr": 0.01,
			"seed": 7,
			"batch_tokens": 8,
			"optimizer": "adamw",
			"weight_decay": 0.0,
			"grad_clip": 1.0
		}
	}`), "distill_student")
	if err != nil {
		panic(err)
	}
	return cfg
}

func writeTinyDistillationTeacherFixture(t *testing.T) tinyDistillationFixture {
	t.Helper()
	dir := t.TempDir()
	teacherConfig := `{
		"name": "distill_teacher",
		"model_dim": 8,
		"vocab_size": 16,
		"seq_len": 4,
		"blocks": [
			{"type": "plain", "heads": 2}
		],
		"training": {
			"steps": 2,
			"lr": 0.01,
			"seed": 11,
			"batch_tokens": 8,
			"optimizer": "adamw",
			"weight_decay": 0.0
		}
	}`
	teacherConfigPath := filepath.Join(dir, "teacher.json")
	if err := os.WriteFile(teacherConfigPath, []byte(teacherConfig), 0o644); err != nil {
		t.Fatalf("WriteFile teacher config: %v", err)
	}
	cfg, err := ParseArchConfig([]byte(teacherConfig), teacherConfigPath)
	if err != nil {
		t.Fatalf("ParseArchConfig teacher: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes teacher: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	teacherWeightsPath := filepath.Join(dir, "teacher.safetensors")
	if err := exportSafetensors(teacherWeightsPath, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors teacher: %v", err)
	}
	return tinyDistillationFixture{
		dir:                dir,
		teacherConfigPath:  teacherConfigPath,
		teacherWeightsPath: teacherWeightsPath,
	}
}
