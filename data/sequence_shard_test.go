package data

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeSequenceShardFixture(t *testing.T, path string, records [][]uint16) {
	t.Helper()
	header := make([]byte, headerInts*4)
	nTokens := 0
	for _, record := range records {
		nTokens += len(record)
	}
	binary.LittleEndian.PutUint32(header[0:4], sequenceShardMagic)
	binary.LittleEndian.PutUint32(header[4:8], sequenceShardVersion)
	binary.LittleEndian.PutUint32(header[8:12], uint32(nTokens))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(records)))
	offsets := make([]byte, (len(records)+1)*8)
	pos := 0
	for i, record := range records {
		binary.LittleEndian.PutUint64(offsets[i*8:], uint64(pos))
		pos += len(record)
	}
	binary.LittleEndian.PutUint64(offsets[len(records)*8:], uint64(pos))
	payload := make([]byte, nTokens*2)
	pos = 0
	for _, record := range records {
		for _, token := range record {
			binary.LittleEndian.PutUint16(payload[pos*2:], token)
			pos++
		}
	}
	if err := os.WriteFile(path, append(append(header, offsets...), payload...), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSequenceManifestFixture(t *testing.T, dir string) {
	t.Helper()
	manifest := DatasetManifest{
		Format: DatasetManifestFormat, Version: DatasetManifestVersion,
		Representation: DatasetRepresentationDiscreteTokens, Modality: "nucleotide",
		VocabSize: 9, TokenDType: DatasetTokenDTypeUint16, ShardFormat: DatasetShardFormatSequenceV1,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2, "mask": 3},
		Artifacts:       DatasetManifestArtifacts{Vocabulary: "nucleotide_vocab.json"},
		Splits:          map[string]DatasetSplit{"train": {Pattern: "train_*.bin", Tokens: 6, Shards: 1, Sequences: 2}},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadSequenceShardRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "train_00000.bin")
	want := [][]uint16{{4, 5, 6, 7}, {8, 4}}
	writeSequenceShardFixture(t, path, want)
	got, err := LoadSequenceShard(path)
	if err != nil {
		t.Fatalf("LoadSequenceShard: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("records=%v want=%v", got, want)
	}
}

func TestSequenceLoaderPacksBoundariesAndMasksCrossContigTargets(t *testing.T) {
	dir := t.TempDir()
	writeSequenceShardFixture(t, filepath.Join(dir, "train_00000.bin"), [][]uint16{{4, 5}, {6, 7, 8}})
	writeSequenceManifestFixture(t, dir)
	loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 42, LoaderOptions{NoShardShuffle: true})
	if err != nil {
		t.Fatalf("NewLoaderWithOptions: %v", err)
	}
	batch, err := loader.NextBatchDetailed(8, 8)
	if err != nil {
		t.Fatalf("NextBatchDetailed: %v", err)
	}
	wantX := []int{1, 4, 5, 2, 1, 6, 7, 2}
	wantY := []int{4, 5, 2, 2, 6, 7, 2, 2}
	wantLoss := []float32{1, 1, 1, 0, 1, 1, 1, 0}
	wantSegments := []int32{0, 0, 0, 0, 1, 1, 1, 1}
	wantEligible := []uint8{0, 1, 1, 0, 0, 1, 1, 0}
	if !reflect.DeepEqual(batch.X, wantX) || !reflect.DeepEqual(batch.Y, wantY) {
		t.Fatalf("x/y=%v/%v want=%v/%v", batch.X, batch.Y, wantX, wantY)
	}
	if !reflect.DeepEqual(batch.LossMask, wantLoss) || !reflect.DeepEqual(batch.SegmentIDs, wantSegments) || !reflect.DeepEqual(batch.MaskEligible, wantEligible) {
		t.Fatalf("metadata loss=%v segments=%v eligible=%v", batch.LossMask, batch.SegmentIDs, batch.MaskEligible)
	}
	if batch.LossMask[3] != 0 || batch.Y[3] == batch.X[4] {
		t.Fatal("EOS position exposes a target from the next contig")
	}
}
