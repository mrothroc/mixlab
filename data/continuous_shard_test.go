package data

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeContinuousTestShard(
	t *testing.T,
	path string,
	records, featureDim int,
	labels []int32,
	frames []float32,
) {
	t.Helper()
	seqLen := 0
	if records > 0 && featureDim > 0 {
		seqLen = len(frames) / (records * featureDim)
	}
	header := make([]byte, headerInts*4)
	values := []int32{
		continuousSequenceShardMagic,
		continuousSequenceShardV1,
		int32(records),
		int32(seqLen),
		int32(featureDim),
		continuousFeatureDTypeFloat32,
		1,
	}
	for i, value := range values {
		binary.LittleEndian.PutUint32(header[i*4:], uint32(value))
	}
	payload := append([]byte(nil), header...)
	for _, label := range labels {
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], uint32(label))
		payload = append(payload, encoded[:]...)
	}
	for _, frame := range frames {
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(frame))
		payload = append(payload, encoded[:]...)
	}
	if err := os.WriteFile(path, payload, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeContinuousTestManifest(t *testing.T, dir string, records, seqLen, featureDim int) {
	t.Helper()
	manifest := DatasetManifest{
		Format: DatasetManifestFormat, Version: DatasetManifestVersion,
		Representation: DatasetRepresentationContinuousFrames,
		Modality:       "signal",
		FeatureDType:   DatasetFeatureDTypeFloat32,
		FeatureDim:     featureDim,
		ShardFormat:    DatasetShardFormatContinuousSequenceV1,
		SequenceLayout: DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:   seqLen,
		Task:           &DatasetTask{Type: DatasetTaskSingleLabelClassification, NumLabels: 2},
		Splits: map[string]DatasetSplit{
			"train": {
				Pattern: "train_*.bin", Frames: int64(records * seqLen), Shards: 1,
				Sequences: int64(records), ClassCounts: map[string]int64{"0": int64((records + 1) / 2), "1": int64(records / 2)},
			},
		},
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DatasetManifestFilename), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadContinuousSequenceShardAndManifestContract(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	frames := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}
	labels := []int32{0, 1, 0}
	writeContinuousTestShard(t, path, 3, 2, labels, frames)
	writeContinuousTestManifest(t, dir, 3, 2, 2)

	shard, err := LoadContinuousSequenceShard(path)
	if err != nil {
		t.Fatal(err)
	}
	if shard.Records != 3 || shard.SeqLen != 2 || shard.FeatureDim != 2 ||
		!reflect.DeepEqual(shard.Labels, labels) || !reflect.DeepEqual(shard.Frames, frames) {
		t.Fatalf("shard=%+v", shard)
	}
	manifest, err := LoadDatasetManifest(filepath.Join(dir, DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.ValidateContinuousFeatures(2, 2, 2); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name                   string
		featureDim, seq, label int
		want                   string
	}{
		{"feature", 1, 2, 2, "feature_dim"},
		{"sequence", 2, 3, 2, "record_seq_len"},
		{"labels", 2, 2, 3, "num_labels"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := manifest.ValidateContinuousFeatures(tt.featureDim, tt.seq, tt.label); err == nil ||
				!strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestContinuousLoaderAndFullClassificationValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	frames := []float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}
	writeContinuousTestShard(t, path, 3, 2, []int32{0, 1, 0}, frames)
	writeContinuousTestManifest(t, dir, 3, 2, 2)
	pattern := filepath.Join(dir, "train_*.bin")

	loader, err := NewLoaderWithOptions(pattern, 7, LoaderOptions{NoShardShuffle: true})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := loader.NextBatchDetailed(4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if batch.X != nil || batch.Y != nil {
		t.Fatalf("continuous batch retained token rows: X=%v Y=%v", batch.X, batch.Y)
	}
	if !reflect.DeepEqual(batch.Frames, frames[:8]) ||
		!reflect.DeepEqual(batch.Labels, []int32{0, 1}) ||
		!reflect.DeepEqual(batch.ValidMask, []float32{1, 1, 1, 1}) {
		t.Fatalf("batch=%+v", batch)
	}

	val, err := NewClassificationValSet(pattern, 0, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	if val.TotalExamples != 3 || val.EvaluatedExamples != 3 || len(val.Batches) != 2 {
		t.Fatalf("validation summary=%+v", val)
	}
	if val.Batches[1].ExampleCount != 1 ||
		!reflect.DeepEqual(val.Batches[1].Frames[:4], frames[8:12]) ||
		!reflect.DeepEqual(val.Batches[1].Frames[4:8], frames[8:12]) {
		t.Fatalf("padded final batch=%+v", val.Batches[1])
	}
}

func TestContinuousShardRejectsMalformedPayloads(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.bin")
	writeContinuousTestShard(t, path, 1, 1, []int32{0}, []float32{1, 2})

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw[:len(raw)-1], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadContinuousSequenceShard(path); err == nil || !strings.Contains(err.Error(), "size mismatch") {
		t.Fatalf("truncated error=%v", err)
	}

	writeContinuousTestShard(t, path, 1, 1, []int32{0}, []float32{1, float32(math.NaN())})
	if _, err := LoadContinuousSequenceShard(path); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("non-finite error=%v", err)
	}

	writeContinuousTestShard(t, path, 1, 1, []int32{0}, []float32{1, 2})
	raw, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, offset := range []int{2 * 4, 3 * 4, 4 * 4} {
		binary.LittleEndian.PutUint32(raw[offset:], uint32(1<<31-1))
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadContinuousSequenceShard(path); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("oversized-shape error=%v", err)
	}
}
