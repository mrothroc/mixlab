package train

import (
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func nativeClassificationTestConfig() *ArchConfig {
	return &ArchConfig{
		Name: "classification", ModelDim: 8, VocabSize: 16, SeqLen: 6,
		TieEmbeddings: true,
		Blocks:        []BlockSpec{{Type: "plain", Heads: 2}},
		Training: TrainingSpec{
			Objective: arch.ObjectiveClassification, BatchTokens: 12, Steps: 2, LR: 1e-3, Seed: 7,
			Classification: &arch.ClassificationSpec{NumLabels: 2, Pooling: arch.ClassificationPoolingLast},
		},
	}
}

func TestPrepareClassificationBatchBuildsPaddingAwareInputs(t *testing.T) {
	cfg := nativeClassificationTestConfig()
	raw := trainBatch{
		x:         []int{1, 4, 5, 2, 0, 0, 1, 6, 7, 8, 2, 0},
		y:         make([]int, 12),
		labels:    []int32{1, 0},
		validMask: []float32{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0},
		segmentIDs: []int32{
			0, 0, 0, 0, 1, 1,
			0, 0, 0, 0, 0, 1,
		},
	}
	got, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveClassification)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.classificationLabels, []int32{1, 0}) {
		t.Fatalf("labels=%v", got.classificationLabels)
	}
	if !reflect.DeepEqual(got.classificationPos, []int32{3, 10}) {
		t.Fatalf("positions=%v want [3 10]", got.classificationPos)
	}
	if !reflect.DeepEqual(got.classificationMask, raw.validMask) || !reflect.DeepEqual(got.segmentIDs, raw.segmentIDs) {
		t.Fatalf("classification mask/segments=%v/%v", got.classificationMask, got.segmentIDs)
	}

	raw.labels[1] = 2
	if _, err := prepareObjectiveBatch(cfg, raw, 0, arch.ObjectiveClassification); err == nil {
		t.Fatal("classification accepted a label outside num_labels")
	}
}

func TestClassificationMetricOracles(t *testing.T) {
	got := classificationMetricsFromPredictions(
		[]int{0, 0, 1, 1},
		[]int{0, 1, 1, 1},
		[]float64{0.1, 0.4, 0.35, 0.8},
		2,
	)
	assertNear(t, "accuracy", got.Accuracy, 0.75)
	assertNear(t, "mcc", got.MCC, 1/math.Sqrt(3))
	assertNear(t, "macro_f1", got.MacroF1, 11.0/15.0)
	if !got.HasAUROC {
		t.Fatal("binary AUROC is unavailable")
	}
	assertNear(t, "auroc", got.AUROC, 0.75)

	auc, ok := binaryAUROC([]int{0, 1}, []float64{0.5, 0.5})
	if !ok {
		t.Fatal("tied-score AUROC is unavailable")
	}
	assertNear(t, "tied auroc", auc, 0.5)
	if _, ok := binaryAUROC([]int{1, 1}, []float64{0.2, 0.8}); ok {
		t.Fatal("single-class AUROC should be unavailable")
	}
}

func TestNativeClassificationHFExportFailsExplicitly(t *testing.T) {
	err := validateHFExportConfig(nativeClassificationTestConfig())
	if err == nil || !strings.Contains(err.Error(), "training.objective") {
		t.Fatalf("error=%v", err)
	}
}

func TestClassificationWarmStartLoadsExactLMPrefix(t *testing.T) {
	cfg := nativeClassificationTestConfig()
	classificationShapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseCfg := *cfg
	baseCfg.Training.Objective = arch.ObjectiveCausal
	baseCfg.Training.Classification = nil
	baseShapes, err := computeWeightShapes(&baseCfg)
	if err != nil {
		t.Fatal(err)
	}
	baseWeights := initWeightData(baseShapes, 19, "", 0)
	path := filepath.Join(t.TempDir(), "pretrained.safetensors")
	if err := exportSafetensors(path, &baseCfg, baseShapes, baseWeights); err != nil {
		t.Fatal(err)
	}
	loaded, fresh, err := loadClassificationWarmStartWeights(
		path, classificationShapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd,
	)
	if err != nil {
		t.Fatal(err)
	}
	if fresh != 2 || len(loaded) != len(baseWeights)+2 {
		t.Fatalf("fresh=%d loaded=%d base=%d", fresh, len(loaded), len(baseWeights))
	}
	for i := range baseWeights {
		if !reflect.DeepEqual(loaded[i], baseWeights[i]) {
			t.Fatalf("warm-start prefix weight %d changed", i)
		}
	}
	for _, value := range loaded[len(loaded)-1] {
		if value != 0 {
			t.Fatalf("classification bias initialized to %g, want zero", value)
		}
	}
}

func assertNear(t *testing.T, name string, got, want float64) {
	t.Helper()
	const tolerance = 1e-12
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s=%g want=%g tolerance=%g", name, got, want, tolerance)
	}
}
