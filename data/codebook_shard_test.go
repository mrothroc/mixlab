package data

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// codebookTestVocabSize is the codebook domain every fixture in this package
// uses; the shard header and manifest must agree on it.
const codebookTestVocabSize = 8

func writeCodebookTestShard(t *testing.T, path string, records, seqLen, codebooks int, labels, lengths, tokens []int32) {
	t.Helper()
	header := make([]int32, headerInts)
	header[0] = codebookSequenceShardMagic
	header[1] = codebookSequenceShardVersion
	header[2] = int32(records)
	header[3] = int32(seqLen)
	header[4] = int32(codebooks)
	header[5] = int32(codebookTestVocabSize)
	header[6] = codebookTokenDTypeInt32
	header[7] = 1
	header[8] = 1
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := file.Close(); err != nil {
			t.Fatalf("close shard: %v", err)
		}
	}()
	for _, values := range [][]int32{header, labels, lengths, tokens} {
		if err := binary.Write(file, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
}

func writeCodebookTestManifest(t *testing.T, dir string, records, seqLen, codebooks int) {
	t.Helper()
	manifest := DatasetManifest{
		Format: DatasetManifestFormat, Version: DatasetManifestVersion,
		Representation: DatasetRepresentationDiscreteCodebooks,
		Modality:       "audio", TokenDType: DatasetTokenDTypeInt32,
		NumCodebooks: codebooks, CodebookVocabSize: codebookTestVocabSize,
		ShardFormat:    DatasetShardFormatCodebookSequenceV1,
		SequenceLayout: DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:   seqLen,
		Task:           &DatasetTask{Type: DatasetTaskSingleLabelClassification, NumLabels: 2},
		Splits: map[string]DatasetSplit{
			"train": {
				Pattern: "train_*.bin", Tokens: int64(records * seqLen * codebooks), Shards: 1,
				Sequences: int64(records), MeanSequenceTokens: float64(seqLen), MaxSequenceTokens: seqLen,
				ClassCounts: map[string]int64{"0": int64((records + 1) / 2), "1": int64(records / 2)},
			},
		},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DatasetManifestFilename), blob, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestCodebookShardLoaderAndClassificationValidation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	tokens := []int32{
		0, 1, 2, 3, 4, 5,
		6, 7, 0, 1, 2, 3,
		4, 5, 6, 7, 1, 0,
	}
	writeCodebookTestShard(t, path, 3, 3, 2, []int32{0, 1, 0}, []int32{3, 2, 1}, tokens)
	writeCodebookTestManifest(t, dir, 3, 3, 2)

	shard, err := LoadCodebookSequenceShard(path)
	if err != nil {
		t.Fatal(err)
	}
	if shard.Records != 3 || shard.SeqLen != 3 || shard.NumCodebooks != 2 || shard.CodebookVocabSize != 8 ||
		!reflect.DeepEqual(shard.Tokens, tokens) || !reflect.DeepEqual(shard.Lengths, []int32{3, 2, 1}) {
		t.Fatalf("shard=%+v", shard)
	}
	manifest, err := LoadDatasetManifest(filepath.Join(dir, DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.ValidateDiscreteCodebooks(2, 8, 3, 2); err != nil {
		t.Fatal(err)
	}

	loader, err := NewLoaderWithOptions(filepath.Join(dir, "train_*.bin"), 7, LoaderOptions{NoShardShuffle: true})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := loader.NextBatchDetailed(6, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(batch.Codebooks, tokens[:12]) || !reflect.DeepEqual(batch.Labels, []int32{0, 1}) ||
		!reflect.DeepEqual(batch.ValidMask, []float32{1, 1, 1, 1, 1, 0}) {
		t.Fatalf("batch=%+v", batch)
	}

	val, err := NewClassificationValSet(filepath.Join(dir, "train_*.bin"), 0, 6, 3)
	if err != nil {
		t.Fatal(err)
	}
	if val.TotalExamples != 3 || val.EvaluatedExamples != 3 || len(val.Batches) != 2 ||
		val.Batches[1].ExampleCount != 1 || len(val.Batches[0].Codebooks) != 12 {
		t.Fatalf("val=%+v", val)
	}
}

func TestCodebookShardRejectsOutOfRangeIDAndBadLength(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "train_00000.bin")
	writeCodebookTestShard(t, path, 1, 2, 2, []int32{0}, []int32{2}, []int32{0, 1, 2, 8})
	if _, err := LoadCodebookSequenceShard(path); err == nil || !strings.Contains(err.Error(), "token[0,1,1]=8 outside [0,8)") {
		t.Fatalf("range error=%v", err)
	}
	writeCodebookTestShard(t, path, 1, 2, 2, []int32{0}, []int32{0}, []int32{0, 1, 2, 3})
	if _, err := LoadCodebookSequenceShard(path); err == nil || !strings.Contains(err.Error(), "length[0]=0") {
		t.Fatalf("length error=%v", err)
	}
}

func TestCodebookManifestContractMismatch(t *testing.T) {
	dir := t.TempDir()
	writeCodebookTestManifest(t, dir, 3, 3, 2)
	manifest, err := LoadDatasetManifest(filepath.Join(dir, DatasetManifestFilename))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		q, v int
		want string
	}{
		{"codebooks", 3, 8, "num_codebooks=2"},
		{"vocabulary", 2, 9, "codebook_vocab_size=8"},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := manifest.ValidateDiscreteCodebooks(test.q, test.v, 3, 2); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v want=%q", err, test.want)
			}
		})
	}
}
