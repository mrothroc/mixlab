package train

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
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

func TestClassificationValidationExcludesPaddedFinalRowsAndWritesPredictions(t *testing.T) {
	cfg := nativeClassificationTestConfig()
	firstLoss, err := classificationCrossEntropy([]float32{3, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	evaluator := &scriptedClassificationEvaluator{
		losses: []float32{float32(firstLoss), 999},
		logits: [][]float32{
			{3, 0, 0, 3},
			{-1, 1, -1, 1},
		},
	}
	valSet := &data.ValSet{
		TotalExamples: 3, EvaluatedExamples: 3,
		Batches: []data.ValBatch{
			classificationValBatch([]int32{0, 1}, 2),
			classificationValBatch([]int32{1, 1}, 1),
		},
	}
	var out bytes.Buffer
	got, err := evaluateClassificationValidationWithEvaluator(cfg, valSet, evaluator, 0, 2, 6, &out)
	if err != nil {
		t.Fatal(err)
	}
	if got.Examples != 3 || got.Accuracy != 1 {
		t.Fatalf("metrics=%+v", got)
	}
	finalLoss, err := classificationCrossEntropy([]float32{-1, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	wantLoss := (2*float64(float32(firstLoss)) + finalLoss) / 3
	if math.Abs(got.Loss-wantLoss) > 1e-7 {
		t.Fatalf("loss=%g want=%g; padded-row GPU loss leaked into metrics", got.Loss, wantLoss)
	}

	dec := json.NewDecoder(&out)
	var records []classificationPredictionRecord
	for {
		var record classificationPredictionRecord
		if err := dec.Decode(&record); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		records = append(records, record)
	}
	if len(records) != 3 || records[2].Index != 2 || records[2].Label != 1 || records[2].Prediction != 1 {
		t.Fatalf("prediction records=%+v", records)
	}
}

func TestClassificationMetricsAreOrderInvariant(t *testing.T) {
	ordered := classificationMetricsFromPredictions(
		[]int{0, 0, 1, 1},
		[]int{0, 1, 0, 1},
		[]float64{0.1, 0.8, 0.2, 0.9},
		2,
	)
	reordered := classificationMetricsFromPredictions(
		[]int{1, 0, 1, 0},
		[]int{1, 1, 0, 0},
		[]float64{0.9, 0.8, 0.2, 0.1},
		2,
	)
	if !reflect.DeepEqual(ordered, reordered) {
		t.Fatalf("metrics changed under record reordering: ordered=%+v reordered=%+v", ordered, reordered)
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

type scriptedClassificationEvaluator struct {
	losses  []float32
	logits  [][]float32
	current int
}

func (e *scriptedClassificationEvaluator) EvaluateObjectiveGPUWithOutputs(
	_ objectiveBatch, _, _ int, _ []string,
) (float32, error) {
	if e.current >= len(e.losses) || e.current >= len(e.logits) {
		return 0, io.EOF
	}
	return e.losses[e.current], nil
}

func (e *scriptedClassificationEvaluator) ReadOutput(_ string, _ []int) ([]float32, error) {
	if e.current >= len(e.logits) {
		return nil, io.EOF
	}
	logits := e.logits[e.current]
	e.current++
	return logits, nil
}

func classificationValBatch(labels []int32, realRows int) data.ValBatch {
	const seqLen = 6
	x := make([]int, len(labels)*seqLen)
	valid := make([]float32, len(x))
	segments := make([]int32, len(x))
	for row := range labels {
		start := row * seqLen
		copy(x[start:start+4], []int{1, 4, 5, 2})
		for pos := start; pos < start+4; pos++ {
			valid[pos] = 1
		}
		for pos := start + 4; pos < start+seqLen; pos++ {
			segments[pos] = 1
		}
	}
	return data.ValBatch{
		X: x, Y: make([]int, len(x)), Labels: labels,
		ValidMask: valid, SegmentIDs: segments, ExampleCount: realRows,
	}
}
