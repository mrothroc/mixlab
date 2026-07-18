package data

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestLoadDatasetManifestAndValidateModelVocab(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, DatasetManifestFilename)
	raw := `{
		"format":" MIXLAB.DATASET ",
		"version":1,
		"representation":" DISCRETE_TOKENS ",
		"modality":" TEXT ",
		"vocab_size":32,
		"token_dtype":" UINT16 ",
		"shard_format":" MIXLAB_TOKEN_SHARD_V1 ",
		"special_token_ids":{"[PAD]":0,"[MASK]":4},
		"artifacts":{"tokenizer":"tokenizer.json"},
		"splits":{"train":{"pattern":"train_*.bin","tokens":100,"shards":2}}
	}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, err := LoadDatasetManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Modality != "text" || manifest.Representation != DatasetRepresentationDiscreteTokens {
		t.Fatalf("manifest normalization failed: %+v", manifest)
	}
	if err := manifest.ValidateModelVocab(32); err != nil {
		t.Fatalf("ValidateModelVocab: %v", err)
	}
	if err := manifest.ValidateModelVocab(31); err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("vocab mismatch error=%v", err)
	}
}

func TestFindDatasetManifestOptionalAndAdjacent(t *testing.T) {
	dir := t.TempDir()
	pattern := filepath.Join(dir, "train_*.bin")
	if err := os.WriteFile(filepath.Join(dir, "train_000.bin"), []byte("placeholder"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, found, err := FindDatasetManifest(pattern); err != nil || found {
		t.Fatalf("missing optional manifest found=%v err=%v", found, err)
	}
	writeDatasetManifestFixture(t, dir, 16)
	manifest, path, found, err := FindDatasetManifest(pattern)
	if err != nil {
		t.Fatal(err)
	}
	if !found || path != filepath.Join(dir, DatasetManifestFilename) || manifest.VocabSize != 16 {
		t.Fatalf("found=%v path=%q manifest=%+v", found, path, manifest)
	}
}

func TestDatasetManifestValidationErrors(t *testing.T) {
	valid := DatasetManifest{
		Format:         DatasetManifestFormat,
		Version:        DatasetManifestVersion,
		Representation: DatasetRepresentationDiscreteTokens,
		Modality:       "text",
		VocabSize:      16,
		TokenDType:     DatasetTokenDTypeUint16,
		ShardFormat:    DatasetShardFormatTokenStreamV1,
		Splits:         map[string]DatasetSplit{"train": {Pattern: "train_*.bin", Tokens: 10, Shards: 1}},
	}
	tests := []struct {
		name    string
		mutate  func(*DatasetManifest)
		wantErr string
	}{
		{name: "format", mutate: func(m *DatasetManifest) { m.Format = "other" }, wantErr: "format"},
		{name: "version", mutate: func(m *DatasetManifest) { m.Version = 2 }, wantErr: "unsupported"},
		{name: "representation", mutate: func(m *DatasetManifest) { m.Representation = "continuous_frames" }, wantErr: "representation"},
		{name: "modality", mutate: func(m *DatasetManifest) { m.Modality = "DNA sequence" }, wantErr: "modality"},
		{name: "vocab", mutate: func(m *DatasetManifest) { m.VocabSize = 65537 }, wantErr: "vocab_size"},
		{name: "dtype", mutate: func(m *DatasetManifest) { m.TokenDType = "int32" }, wantErr: "token_dtype"},
		{name: "shard format", mutate: func(m *DatasetManifest) { m.ShardFormat = "v2" }, wantErr: "shard_format"},
		{name: "special id", mutate: func(m *DatasetManifest) { m.SpecialTokenIDs = map[string]int{"mask": 16} }, wantErr: "outside"},
		{name: "duplicate special id", mutate: func(m *DatasetManifest) { m.SpecialTokenIDs = map[string]int{"mask": 4, "pad": 4} }, wantErr: "both use id"},
		{name: "absolute artifact", mutate: func(m *DatasetManifest) { m.Artifacts.Tokenizer = "/tmp/tokenizer.json" }, wantErr: "relative"},
		{name: "escaping artifact", mutate: func(m *DatasetManifest) { m.Artifacts.Vocabulary = "../vocab.json" }, wantErr: "escapes"},
		{name: "no splits", mutate: func(m *DatasetManifest) { m.Splits = nil }, wantErr: "at least one"},
		{name: "empty split pattern", mutate: func(m *DatasetManifest) { m.Splits = map[string]DatasetSplit{"train": {Tokens: 10, Shards: 1}} }, wantErr: "must not be empty"},
		{name: "token shard mismatch", mutate: func(m *DatasetManifest) {
			m.Splits = map[string]DatasetSplit{"train": {Pattern: "train.bin", Tokens: 10}}
		}, wantErr: "zero shards"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := valid
			got.SpecialTokenIDs = nil
			got.Artifacts = DatasetManifestArtifacts{}
			got.Splits = map[string]DatasetSplit{"train": {Pattern: "train_*.bin", Tokens: 10, Shards: 1}}
			tt.mutate(&got)
			if err := got.Validate(); err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error=%v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestLoadDatasetManifestRejectsUnknownFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, DatasetManifestFilename)
	raw := `{"format":"mixlab.dataset","version":1,"representation":"discrete_tokens","modality":"text","vocab_size":8,"token_dtype":"uint16","shard_format":"mixlab_token_shard_v1","splits":{"train":{"pattern":"train.bin","tokens":1,"shards":1}},"surprise":true}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDatasetManifest(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("error=%v", err)
	}
}

func TestLoadDatasetManifestRejectsTrailingJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeDatasetManifestFixture(t, dir, 8)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(` {}`); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDatasetManifest(path); err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
		t.Fatalf("error=%v", err)
	}
}

func writeDatasetManifestFixture(t *testing.T, dir string, vocabSize int) string {
	t.Helper()
	path := filepath.Join(dir, DatasetManifestFilename)
	raw := `{"format":"mixlab.dataset","version":1,"representation":"discrete_tokens","modality":"text","vocab_size":` + strconv.Itoa(vocabSize) + `,"token_dtype":"uint16","shard_format":"mixlab_token_shard_v1","splits":{"train":{"pattern":"train_*.bin","tokens":1,"shards":1}}}`
	if err := os.WriteFile(path, []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
