package data

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestCodebookLengthBucketsSliceMaskAndKeepPartialRows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	lengths := []int32{1, 2, 3, 5, 6}
	labels := []int32{0, 1, 0, 1, 0}
	tokens := make([]int32, 5*6)
	for i := range tokens {
		tokens[i] = int32(i % 8)
	}
	writeCodebookTestShard(t, path, 5, 6, 1, labels, lengths, tokens)
	writeCodebookTestManifest(t, dir, 5, 6, 1)
	loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 9, LoaderOptions{
		NoShardShuffle: true, LengthBuckets: []int{2, 4, 6},
	})
	if err != nil {
		t.Fatal(err)
	}
	first, err := loader.NextBatchDetailed(8, 6)
	if err != nil {
		t.Fatal(err)
	}
	if first.SeqLen != 2 || first.BatchSize != 4 || first.ExampleCount != 2 ||
		!reflect.DeepEqual(first.ExampleMask, []float32{1, 1, 0, 0}) ||
		!reflect.DeepEqual(first.ValidMask, []float32{1, 0, 1, 1, 1, 0, 1, 0}) {
		t.Fatalf("first bucket=%+v", first)
	}
	if !reflect.DeepEqual(first.Codebooks[:4], []int32{0, 1, 6, 7}) {
		t.Fatalf("sliced codebooks=%v", first.Codebooks)
	}
	second, err := loader.NextBatchDetailed(8, 6)
	if err != nil {
		t.Fatal(err)
	}
	if second.SeqLen != 4 || second.BatchSize != 2 || second.ExampleCount != 1 || !reflect.DeepEqual(second.ExampleMask, []float32{1, 0}) {
		t.Fatalf("second bucket=%+v", second)
	}

	val, err := NewClassificationValSetWithOptions(filepath.Join(dir, "train_*.bin"), 0, 8, 6, LoaderOptions{LengthBuckets: []int{2, 4, 6}})
	if err != nil {
		t.Fatal(err)
	}
	if val.TotalExamples != 5 || val.EvaluatedExamples != 5 || len(val.Batches) != 4 {
		t.Fatalf("bucketed validation=%+v", val)
	}
}

func TestLengthBucketLoaderIsSeedDeterministic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	lengths := []int32{1, 2, 3, 4, 5, 6}
	labels := []int32{0, 1, 0, 1, 0, 1}
	tokens := make([]int32, 6*6)
	writeCodebookTestShard(t, path, 6, 6, 1, labels, lengths, tokens)
	writeCodebookTestManifest(t, dir, 6, 6, 1)
	newLoader := func() *Loader {
		loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 77, LoaderOptions{LengthBuckets: []int{2, 4, 6}})
		if err != nil {
			t.Fatal(err)
		}
		return loader
	}
	a, b := newLoader(), newLoader()
	for step := 0; step < 8; step++ {
		gotA, errA := a.NextBatchDetailed(8, 6)
		gotB, errB := b.NextBatchDetailed(8, 6)
		if errA != nil || errB != nil || !reflect.DeepEqual(gotA, gotB) {
			t.Fatalf("step %d deterministic batches differ: a=%+v err=%v b=%+v err=%v", step, gotA, errA, gotB, errB)
		}
	}
}

func TestLengthBucketLoaderRejectsOversizedRecordAtStartup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	writeCodebookTestShard(t, path, 2, 6, 1, []int32{0, 1}, []int32{2, 6}, make([]int32, 12))
	writeCodebookTestManifest(t, dir, 2, 6, 1)
	_, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 1, LoaderOptions{LengthBuckets: []int{2, 4}})
	if err == nil || !strings.Contains(err.Error(), "train_00000.bin[1]") || !strings.Contains(err.Error(), "valid length 6") || !strings.Contains(err.Error(), "value 4") {
		t.Fatalf("overflow error=%v", err)
	}
}

func writeContinuousV2TestShard(t *testing.T, path string, records, seqLen, featureDim int, labels, lengths []int32, frames []float32) {
	t.Helper()
	header := make([]int32, headerInts)
	header[0], header[1] = continuousSequenceShardMagic, continuousSequenceShardVersion
	header[2], header[3], header[4] = int32(records), int32(seqLen), int32(featureDim)
	header[5], header[6], header[7] = continuousFeatureDTypeFloat32, 1, 1
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	for _, values := range [][]int32{header, labels, lengths} {
		if err := binary.Write(file, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range frames {
		if err := binary.Write(file, binary.LittleEndian, math.Float32bits(value)); err != nil {
			t.Fatal(err)
		}
	}
}

func TestContinuousV2LengthBucketsAndV1Compatibility(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	frames := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	writeContinuousV2TestShard(t, path, 2, 4, 1, []int32{0, 1}, []int32{2, 4}, frames)
	writeContinuousTestManifest(t, dir, 2, 4, 1)
	shard, err := LoadContinuousSequenceShard(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(shard.Lengths, []int32{2, 4}) {
		t.Fatalf("v2 lengths=%v", shard.Lengths)
	}
	loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 1, LoaderOptions{NoShardShuffle: true, LengthBuckets: []int{2, 4}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := loader.NextBatchDetailed(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if batch.SeqLen != 2 || batch.BatchSize != 2 || batch.ExampleCount != 1 || !reflect.DeepEqual(batch.ValidMask, []float32{1, 1, 1, 1}) {
		t.Fatalf("continuous bucket=%+v", batch)
	}

	v1 := filepath.Join(dir, "v1.bin")
	writeContinuousTestShard(t, v1, 2, 1, []int32{0, 1}, frames)
	legacy, err := LoadContinuousSequenceShard(v1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(legacy.Lengths, []int32{4, 4}) {
		t.Fatalf("v1 lengths=%v want full width", legacy.Lengths)
	}
}
