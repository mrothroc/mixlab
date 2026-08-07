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

func continuousTrainTestConfig(t *testing.T) *ArchConfig {
	t.Helper()
	raw := []byte(`{
		"name":"continuous_train_test",
		"model_dim":8,
		"seq_len":4,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":2,"bias":true,"norm":"layernorm"},
		"blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":3,"pooling":"mean"},
			"batch_tokens":8,
			"steps":2,
			"lr":0.001
		}
	}`)
	cfg, err := ParseArchConfig(raw, "continuous_train_test")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestPrepareContinuousClassificationObjectiveBatch(t *testing.T) {
	cfg := continuousTrainTestConfig(t)
	frames := make([]float32, 8*2)
	for i := range frames {
		frames[i] = float32(i)
	}
	prepared, err := prepareObjectiveBatch(cfg, trainBatch{
		frames: frames,
		labels: []int32{2, 1},
		validMask: []float32{
			1, 1, 1, 1,
			1, 1, 1, 1,
		},
	}, 0, arch.ObjectiveClassification)
	if err != nil {
		t.Fatal(err)
	}
	if prepared.x != nil || prepared.y != nil || prepared.unmaskedX != nil {
		t.Fatalf("continuous objective retained token inputs: %+v", prepared)
	}
	if !reflect.DeepEqual(prepared.frames, frames) ||
		!reflect.DeepEqual(prepared.classificationLabels, []int32{2, 1}) ||
		!reflect.DeepEqual(prepared.classificationMask, []float32{1, 1, 1, 1, 1, 1, 1, 1}) {
		t.Fatalf("prepared=%+v", prepared)
	}

	bad := trainBatch{
		frames: frames[:len(frames)-1], labels: []int32{2, 1},
		validMask: make([]float32, 8),
	}
	if _, err := prepareObjectiveBatch(cfg, bad, 0, arch.ObjectiveClassification); err == nil ||
		!strings.Contains(err.Error(), "continuous_frames") {
		t.Fatalf("short-frame error=%v", err)
	}
}

func TestContinuousManifestMismatchFailsBeforeTrainerSetup(t *testing.T) {
	dir := t.TempDir()
	pattern := filepath.Join(dir, "train_*.bin")
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationContinuousFrames,
		Modality:       "signal",
		FeatureDType:   data.DatasetFeatureDTypeFloat32,
		FeatureDim:     3,
		ShardFormat:    data.DatasetShardFormatContinuousSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow,
		RecordSeqLen:   4,
		Task:           &data.DatasetTask{Type: data.DatasetTaskSingleLabelClassification, NumLabels: 3},
		Splits: map[string]data.DatasetSplit{
			"train": {
				Pattern: "train_*.bin", Frames: 8, Shards: 1, Sequences: 2,
				ClassCounts: map[string]int64{"0": 1, "1": 1, "2": 0},
			},
		},
	}
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := continuousTrainTestConfig(t)
	if err := configureDatasetForTraining(cfg, pattern, "train"); err == nil ||
		!strings.Contains(err.Error(), "feature_dim") {
		t.Fatalf("manifest mismatch error=%v", err)
	}
}

func TestContinuousInputRejectsUnsupportedHFExportStack(t *testing.T) {
	err := validateHFExportConfig(continuousTrainTestConfig(t))
	if err == nil || !strings.Contains(err.Error(), "sequential s4d-only stacks") {
		t.Fatalf("HF export error=%v", err)
	}
}
