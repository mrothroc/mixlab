package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestPrepareLengthBucketClassificationBatchCarriesExampleMask(t *testing.T) {
	cfg := discreteCodebookTrainTestConfig(t)
	cfg.Training.LengthBuckets = []int{2, 4}
	codes := make([]int32, 4*2*cfg.InputAdapter.NumCodebooks)
	prepared, err := prepareObjectiveBatchWithSeqLen(cfg, trainBatch{
		codebooks:   codes,
		labels:      []int32{0, 1, 2, 0},
		validMask:   []float32{1, 0, 1, 1, 1, 0, 1, 1},
		exampleMask: []float32{1, 1, 1, 0},
	}, 0, arch.ObjectiveClassification, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prepared.classificationRowMask, []float32{1, 1, 1, 0}) ||
		!reflect.DeepEqual(prepared.classificationPos, []int32{0, 3, 4, 7}) ||
		!reflect.DeepEqual(prepared.segmentIDs, []int32{0, 1, 0, 0, 0, 1, 0, 0}) {
		t.Fatalf("prepared=%+v", prepared)
	}
}

func TestPrepareLengthBucketClassificationRejectsInvalidExampleMask(t *testing.T) {
	cfg := discreteCodebookTrainTestConfig(t)
	cfg.Training.LengthBuckets = []int{2, 4}
	_, err := prepareObjectiveBatchWithSeqLen(cfg, trainBatch{
		codebooks:   make([]int32, 4*2*cfg.InputAdapter.NumCodebooks),
		labels:      []int32{0, 1, 2, 0},
		validMask:   make([]float32, 8),
		exampleMask: []float32{1, 1, 0.5, 0},
	}, 0, arch.ObjectiveClassification, 2)
	if err == nil {
		t.Fatal("invalid classification example mask was accepted")
	}
}

func TestPrepareLengthBucketUsesEffectiveTokensBelowCeiling(t *testing.T) {
	cfg := discreteCodebookTrainTestConfig(t)
	cfg.Training.BatchTokens = 10
	cfg.Training.LengthBuckets = []int{2, 4}
	prepared, err := prepareObjectiveBatchWithShape(cfg, trainBatch{
		codebooks: make([]int32, 2*4*cfg.InputAdapter.NumCodebooks),
		labels:    []int32{0, 1}, validMask: []float32{1, 1, 1, 1, 1, 1, 1, 1}, exampleMask: []float32{1, 1},
	}, 0, arch.ObjectiveClassification, 2, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.codebooks) != 16 || len(prepared.classificationMask) != 8 {
		t.Fatalf("prepared effective shape codebooks=%d mask=%d", len(prepared.codebooks), len(prepared.classificationMask))
	}
}

func TestSingleFullWidthLengthBucketUsesLegacyLoaderOptions(t *testing.T) {
	cfg := discreteCodebookTrainTestConfig(t)
	cfg.Training.LengthBuckets = []int{cfg.SeqLen}
	if cfg.Training.LengthBucketsChangeShape(cfg.SeqLen) {
		t.Fatal("single full-width bucket should not change the batch shape")
	}
	if got := effectiveLoaderOptions(cfg).LengthBuckets; len(got) != 0 {
		t.Fatalf("legacy no-op path passed length buckets to loader: %v", got)
	}
}
