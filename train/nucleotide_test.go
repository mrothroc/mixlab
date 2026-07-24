package train

import (
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func nucleotideObjectiveConfig(objective string, reverseProb float64) *ArchConfig {
	return &ArchConfig{
		Name: "nucleotide-test", ModelDim: 8, VocabSize: 9, SeqLen: 5,
		Training: TrainingSpec{
			Objective: objective, BatchTokens: 5, Seed: 17,
			MLMMaskProb: 1, MLMMaskTokenID: 3, MLMMaskTokenProb: 1,
			ReverseComplementProb:  reverseProb,
			DatasetSequencePacking: true, DatasetBOSID: 1, DatasetEOSID: 2, DatasetPADID: 0,
			DatasetNucleotideAlphabet:   "dna",
			DatasetNucleotideComplement: []int{0, 1, 2, 3, 7, 6, 5, 4, 8},
			DatasetTokenEligible:        []uint8{0, 0, 0, 0, 1, 1, 1, 1, 1},
		},
	}
}

func nucleotideTrainBatch() trainBatch {
	return trainBatch{
		x: []int{1, 4, 5, 5, 2}, y: []int{4, 5, 5, 2, 2},
		lossMask:     []float32{1, 1, 1, 1, 0},
		segmentIDs:   []int32{0, 0, 0, 0, 0},
		maskEligible: []uint8{0, 1, 1, 1, 0},
	}
}

func TestReverseComplementAugmentationGoldenAndDeterministic(t *testing.T) {
	if reverseComplementRNGSalt == wordStructuralRNGSalt {
		t.Fatal("reverse-complement and word-structural objectives must use independent RNG domains")
	}
	cfg := nucleotideObjectiveConfig(arch.ObjectiveCausal, 1)
	first, err := prepareObjectiveBatch(cfg, nucleotideTrainBatch(), 9, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch: %v", err)
	}
	second, err := prepareObjectiveBatch(cfg, nucleotideTrainBatch(), 9, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch repeat: %v", err)
	}
	wantX := []int{1, 6, 6, 7, 2}
	wantY := []int{6, 6, 7, 2, 2}
	if !reflect.DeepEqual(first.x, wantX) || !reflect.DeepEqual(first.y, wantY) {
		t.Fatalf("reverse complement x/y=%v/%v want=%v/%v", first.x, first.y, wantX, wantY)
	}
	if !reflect.DeepEqual(first.x, second.x) || !reflect.DeepEqual(first.y, second.y) {
		t.Fatal("same seed and step produced different reverse complements")
	}
	if !reflect.DeepEqual(first.lossMask, []float32{1, 1, 1, 1, 0}) || !reflect.DeepEqual(first.segmentIDs, []int32{0, 0, 0, 0, 0}) {
		t.Fatalf("metadata changed loss=%v segments=%v", first.lossMask, first.segmentIDs)
	}
}

func TestReverseComplementDisabledIsExactAndValidationSkipsIt(t *testing.T) {
	batch := nucleotideTrainBatch()
	cfg := nucleotideObjectiveConfig(arch.ObjectiveCausal, 0)
	got, err := prepareObjectiveBatch(cfg, batch, 3, arch.ObjectiveCausal)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.x, batch.x) || !reflect.DeepEqual(got.y, batch.y) {
		t.Fatalf("disabled augmentation changed batch: %v/%v", got.x, got.y)
	}
	cfg.Training.ReverseComplementProb = 1
	batch.disableAugmentation = true
	got, err = prepareObjectiveBatch(cfg, batch, 3, arch.ObjectiveCausal)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.x, batch.x) {
		t.Fatalf("validation batch was augmented: %v", got.x)
	}
}

func TestReverseComplementDoesNotCrossBatchRowsWithRepeatedSegmentIDs(t *testing.T) {
	cfg := nucleotideObjectiveConfig(arch.ObjectiveCausal, 1)
	cfg.Training.BatchTokens = 10
	batch := trainBatch{
		x: []int{
			1, 4, 5, 6, 2,
			1, 7, 4, 5, 2,
		},
		y: []int{
			4, 5, 6, 2, 2,
			7, 4, 5, 2, 2,
		},
		lossMask: []float32{
			1, 1, 1, 1, 0,
			1, 1, 1, 1, 0,
		},
		segmentIDs: []int32{
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
		},
		maskEligible: []uint8{
			0, 1, 1, 1, 0,
			0, 1, 1, 1, 0,
		},
	}
	got, err := prepareObjectiveBatch(cfg, batch, 4, arch.ObjectiveCausal)
	if err != nil {
		t.Fatal(err)
	}
	wantX := []int{
		1, 5, 6, 7, 2,
		1, 6, 7, 4, 2,
	}
	if !reflect.DeepEqual(got.x, wantX) {
		t.Fatalf("reverse complement crossed row boundary: x=%v want=%v", got.x, wantX)
	}
	wantY := []int{
		5, 6, 7, 2, 2,
		6, 7, 4, 2, 2,
	}
	if !reflect.DeepEqual(got.y, wantY) {
		t.Fatalf("reverse-complement targets=%v want=%v", got.y, wantY)
	}
}

func TestAttachRCEquivariantInputsReversesOnlyBiologicalPositions(t *testing.T) {
	cfg := nucleotideObjectiveConfig(arch.ObjectiveMLM, 0)
	cfg.RCEquivariant = true
	cfg.SeqLen = 7
	cfg.Training.BatchTokens = 7
	batch := trainBatch{
		x:            []int{1, 4, 5, 6, 7, 2, 0}, // BOS A C G T EOS PAD
		y:            make([]int, 7),
		maskEligible: []uint8{0, 1, 1, 1, 1, 0, 0},
		segmentIDs:   []int32{0, 0, 0, 0, 0, 0, 0},
	}
	prepared := objectiveBatch{x: append([]int(nil), batch.x...)}
	if err := attachRCEquivariantInputs(cfg, batch, &prepared, 7, 7); err != nil {
		t.Fatal(err)
	}
	// Reverse(A C G T) then complement -> A C G T for this palindrome.
	wantTokens := []int{1, 4, 5, 6, 7, 2, 0}
	wantAlignment := []int32{0, 4, 3, 2, 1, 5, 6}
	if !reflect.DeepEqual(prepared.rcTokens, wantTokens) {
		t.Fatalf("rc tokens=%v want=%v", prepared.rcTokens, wantTokens)
	}
	if !reflect.DeepEqual(prepared.rcAlignmentPositions, wantAlignment) {
		t.Fatalf("alignment=%v want=%v", prepared.rcAlignmentPositions, wantAlignment)
	}
}

func TestAttachRCEquivariantInputsStaysWithinPackedSegments(t *testing.T) {
	cfg := nucleotideObjectiveConfig(arch.ObjectiveMLM, 0)
	cfg.RCEquivariant = true
	cfg.SeqLen = 8
	cfg.Training.BatchTokens = 8
	batch := trainBatch{
		x:            []int{1, 4, 5, 2, 1, 6, 7, 2},
		y:            make([]int, 8),
		maskEligible: []uint8{0, 1, 1, 0, 0, 1, 1, 0},
		segmentIDs:   []int32{0, 0, 0, 0, 1, 1, 1, 1},
	}
	prepared := objectiveBatch{x: append([]int(nil), batch.x...)}
	if err := attachRCEquivariantInputs(cfg, batch, &prepared, 8, 8); err != nil {
		t.Fatal(err)
	}
	// AC -> GT and GT -> AC. BOS/EOS remain fixed in each packed segment.
	want := []int{1, 6, 7, 2, 1, 4, 5, 2}
	if !reflect.DeepEqual(prepared.rcTokens, want) {
		t.Fatalf("rc tokens=%v want=%v", prepared.rcTokens, want)
	}
	wantAlignment := []int32{0, 2, 1, 3, 4, 6, 5, 7}
	if !reflect.DeepEqual(prepared.rcAlignmentPositions, wantAlignment) {
		t.Fatalf("alignment=%v want=%v", prepared.rcAlignmentPositions, wantAlignment)
	}
}

func TestReverseComplementAugmentationSupportsLabeledRecordClassification(t *testing.T) {
	cfg := nucleotideObjectiveConfig(arch.ObjectiveClassification, 1)
	cfg.Training.DatasetSequencePacking = false
	cfg.Training.DatasetRecordFraming = true
	cfg.Training.Classification = &arch.ClassificationSpec{NumLabels: 2, Pooling: arch.ClassificationPoolingMean}
	batch := trainBatch{
		x:            []int{1, 4, 4, 5, 2},
		y:            make([]int, 5),
		segmentIDs:   []int32{0, 0, 0, 0, 0},
		maskEligible: []uint8{0, 1, 1, 1, 0},
		labels:       []int32{1},
		validMask:    []float32{1, 1, 1, 1, 1},
	}
	got, err := maybeApplyReverseComplement(cfg, batch, 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 6, 7, 7, 2}
	if !reflect.DeepEqual(got.x, want) {
		t.Fatalf("classification RC augmentation=%v want=%v", got.x, want)
	}
	if !reflect.DeepEqual(got.labels, batch.labels) || !reflect.DeepEqual(got.validMask, batch.validMask) {
		t.Fatal("classification RC augmentation changed labels or validity mask")
	}
}

func TestNucleotideMLMMasksOnlyBiologicalTokens(t *testing.T) {
	cfg := nucleotideObjectiveConfig(arch.ObjectiveMLM, 0)
	got, err := prepareObjectiveBatch(cfg, nucleotideTrainBatch(), 0, arch.ObjectiveMLM)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.lossMask, []float32{0, 1, 1, 1, 0}) {
		t.Fatalf("lossMask=%v", got.lossMask)
	}
	if !reflect.DeepEqual(got.x, []int{1, 3, 3, 3, 2}) {
		t.Fatalf("masked x=%v", got.x)
	}
	if !reflect.DeepEqual(got.y, nucleotideTrainBatch().x) {
		t.Fatalf("MLM targets=%v", got.y)
	}
}
