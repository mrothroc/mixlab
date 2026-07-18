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
