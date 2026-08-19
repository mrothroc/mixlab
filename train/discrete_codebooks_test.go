package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func discreteCodebookTrainTestConfig(t *testing.T) *ArchConfig {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(`{
		"name":"codebook_train_test",
		"model_dim":6,
		"seq_len":4,
		"positional_embedding":"none",
		"input_adapter":{"kind":"discrete_codebooks","num_codebooks":2,"codebook_vocab_size":8,"fusion":"attention_mlp","fusion_hidden_dim":6,"norm":"none"},
		"blocks":[{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":3,"pooling":"mean","bias":false},
			"batch_tokens":8,
			"steps":2,
			"lr":0.001
		}
	}`), "codebook_train_test")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestPrepareDiscreteCodebookClassificationObjectiveBatch(t *testing.T) {
	cfg := discreteCodebookTrainTestConfig(t)
	codes := []int32{
		0, 1, 2, 3, 4, 5, 6, 7,
		7, 6, 5, 4, 3, 2, 1, 0,
	}
	prepared, err := prepareObjectiveBatch(cfg, trainBatch{
		codebooks: codes,
		labels:    []int32{2, 1},
		validMask: []float32{1, 1, 1, 0, 1, 1, 0, 0},
	}, 0, arch.ObjectiveClassification)
	if err != nil {
		t.Fatal(err)
	}
	if prepared.x != nil || prepared.y != nil || prepared.frames != nil ||
		!reflect.DeepEqual(prepared.codebooks, codes) ||
		!reflect.DeepEqual(prepared.classificationLabels, []int32{2, 1}) ||
		!reflect.DeepEqual(prepared.classificationPos, []int32{2, 5}) {
		t.Fatalf("prepared=%+v", prepared)
	}
	if _, err := prepareObjectiveBatch(cfg, trainBatch{
		codebooks: codes[:len(codes)-1], labels: []int32{2, 1}, validMask: make([]float32, 8),
	}, 0, arch.ObjectiveClassification); err == nil || !strings.Contains(err.Error(), "codebook_tokens") {
		t.Fatalf("short-codebook error=%v", err)
	}
}

func TestDiscreteCodebookManifestMismatchFailsBeforeTrainerSetup(t *testing.T) {
	dir := t.TempDir()
	pattern := filepath.Join(dir, "train_*.bin")
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteCodebooks,
		Modality:       "audio", TokenDType: data.DatasetTokenDTypeInt32,
		NumCodebooks: 3, CodebookVocabSize: 8,
		ShardFormat:    data.DatasetShardFormatCodebookSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:   4,
		Task:           &data.DatasetTask{Type: data.DatasetTaskSingleLabelClassification, NumLabels: 3},
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: 24, Shards: 1, Sequences: 2, MeanSequenceTokens: 4, MaxSequenceTokens: 4, ClassCounts: map[string]int64{"0": 1, "1": 1, "2": 0}},
		},
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := discreteCodebookTrainTestConfig(t)
	if err := configureDatasetForTraining(cfg, pattern, "train"); err == nil ||
		!strings.Contains(err.Error(), "num_codebooks=3") || !strings.Contains(err.Error(), "input_adapter.num_codebooks=2") {
		t.Fatalf("manifest mismatch error=%v", err)
	}
}

func TestDiscreteCodebooksRejectHFExport(t *testing.T) {
	err := validateHFExportConfig(discreteCodebookTrainTestConfig(t))
	if err == nil || !strings.Contains(err.Error(), "discrete_codebooks") || !strings.Contains(err.Error(), "native") {
		t.Fatalf("HF export error=%v", err)
	}
}
