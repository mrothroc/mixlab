package data

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestNewClassificationValSetTraversesFullSplitAndPadsFinalBatch(t *testing.T) {
	dir := t.TempDir()
	writeLabeledSequenceShardFixture(
		t, filepath.Join(dir, "val_00000.bin"),
		[][]uint16{{4}, {5, 6}, {7}}, []int32{0, 1, 0},
	)
	writeLabeledSequenceShardFixture(
		t, filepath.Join(dir, "val_00001.bin"),
		[][]uint16{{8, 9}, {10}}, []int32{1, 1},
	)
	writeClassificationValManifest(t, dir, 7, 5, 2, map[string]int64{"0": 2, "1": 3})

	got, err := NewClassificationValSet(filepath.Join(dir, "val_*.bin"), 0, 10, 5)
	if err != nil {
		t.Fatal(err)
	}
	if got.TotalExamples != 5 || got.EvaluatedExamples != 5 || len(got.Batches) != 3 {
		t.Fatalf("validation set totals=%d/%d batches=%d, want 5/5/3", got.EvaluatedExamples, got.TotalExamples, len(got.Batches))
	}
	if counts := []int{got.Batches[0].ExampleCount, got.Batches[1].ExampleCount, got.Batches[2].ExampleCount}; !reflect.DeepEqual(counts, []int{2, 2, 1}) {
		t.Fatalf("batch example counts=%v", counts)
	}
	var labels []int32
	for _, batch := range got.Batches {
		labels = append(labels, batch.Labels[:batch.ExampleCount]...)
	}
	if !reflect.DeepEqual(labels, []int32{0, 1, 0, 1, 1}) {
		t.Fatalf("labels=%v", labels)
	}
	last := got.Batches[2]
	if !reflect.DeepEqual(last.X[:5], last.X[5:]) || last.Labels[0] != last.Labels[1] {
		t.Fatalf("partial batch padding did not duplicate its real row: x=%v labels=%v", last.X, last.Labels)
	}
}

func TestNewClassificationValSetExplicitBatchCapReportsSplitTotal(t *testing.T) {
	dir := t.TempDir()
	writeLabeledSequenceShardFixture(
		t, filepath.Join(dir, "val_00000.bin"),
		[][]uint16{{4}, {5}, {6}, {7}, {8}}, []int32{0, 1, 0, 1, 0},
	)
	writeClassificationValManifest(t, dir, 5, 5, 1, map[string]int64{"0": 3, "1": 2})

	got, err := NewClassificationValSet(filepath.Join(dir, "val_*.bin"), 2, 10, 5)
	if err != nil {
		t.Fatal(err)
	}
	if got.TotalExamples != 5 || got.EvaluatedExamples != 4 || len(got.Batches) != 2 {
		t.Fatalf("validation set totals=%d/%d batches=%d, want 4/5/2", got.EvaluatedExamples, got.TotalExamples, len(got.Batches))
	}
}

func writeClassificationValManifest(t *testing.T, dir string, tokens, sequences int64, shards int, classCounts map[string]int64) {
	t.Helper()
	manifest := DatasetManifest{
		Format: DatasetManifestFormat, Version: DatasetManifestVersion,
		Representation: DatasetRepresentationDiscreteTokens,
		Modality:       "text", VocabSize: 32, TokenDType: DatasetTokenDTypeUint16,
		ShardFormat:    DatasetShardFormatLabeledSequenceV1,
		SequenceLayout: DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:   5,
		SpecialTokenIDs: map[string]int{
			"pad": 0, "bos": 1, "eos": 2,
		},
		Task: &DatasetTask{Type: DatasetTaskSingleLabelClassification, NumLabels: 2},
		Splits: map[string]DatasetSplit{
			"val": {
				Pattern: "val_*.bin", Tokens: tokens, Shards: shards, Sequences: sequences,
				MaxSequenceTokens: 2, ClassCounts: classCounts,
			},
		},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}
