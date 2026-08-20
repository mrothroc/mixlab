//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestNewBobClassificationTrainingUsesFullValidationAndCheckpointsLiveLR(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	dir := t.TempDir()
	records := [][]uint16{{3, 3}, {4, 4}, {3, 4}, {4, 3}}
	labels := []int32{0, 1, 0, 1}
	writeNewBobLabeledSequenceShard(t, filepath.Join(dir, "train_00000.bin"), records, labels)
	writeNewBobLabeledSequenceShard(t, filepath.Join(dir, "val_00000.bin"), records, labels)
	writeNewBobDatasetManifest(t, dir, records, labels)

	cfg, err := ParseArchConfig([]byte(`{
		"name":"newbob_mlx_smoke",
		"model_dim":8,
		"vocab_size":8,
		"seq_len":4,
		"tie_embeddings":true,
		"blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw",
			"steps":2,
			"batch_tokens":8,
			"lr":0.001,
			"lr_schedule":"newbob",
			"newbob":{"annealing_factor":0.5,"improvement_threshold":2,"patient":0,"metric":"val_error_rate"},
			"val_every_steps":1,
			"val_examples":0,
			"weight_decay":0,
			"seed":17
		}
	}`), "newbob_mlx_smoke")
	if err != nil {
		t.Fatal(err)
	}
	checkpointDir := filepath.Join(dir, "checkpoints")
	result, err := runTrain(cfg, filepath.Join(dir, "train_*.bin"), TrainOptions{
		CheckpointDir: checkpointDir, CheckpointEvery: 1, LogEvery: 100,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasValLoss || math.IsNaN(result.LastValLoss) || math.IsInf(result.LastValLoss, 0) {
		t.Fatalf("validation result=%+v", result)
	}
	manifest, err := resolveResumeManifest(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Schedule.NewBob == nil || manifest.Schedule.NewBob.Observations != 2 {
		t.Fatalf("checkpoint NewBob state=%+v", manifest.Schedule.NewBob)
	}
	if got, want := manifest.Schedule.NewBob.CurrentLR, float32(0.0005); math.Abs(float64(got-want)) > 1e-9 {
		t.Fatalf("checkpoint current LR=%g want=%g", got, want)
	}
}

func writeNewBobLabeledSequenceShard(t *testing.T, path string, records [][]uint16, labels []int32) {
	t.Helper()
	const headerBytes = 256 * 4
	header := make([]byte, headerBytes)
	tokens := 0
	for _, record := range records {
		tokens += len(record)
	}
	binary.LittleEndian.PutUint32(header[0:4], 20260724)
	binary.LittleEndian.PutUint32(header[4:8], 1)
	binary.LittleEndian.PutUint32(header[8:12], uint32(tokens))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(records)))
	offsets := make([]byte, (len(records)+1)*8)
	labelBytes := make([]byte, len(labels)*4)
	payload := make([]byte, tokens*2)
	position := 0
	for i, record := range records {
		binary.LittleEndian.PutUint64(offsets[i*8:], uint64(position))
		binary.LittleEndian.PutUint32(labelBytes[i*4:], uint32(labels[i]))
		for _, token := range record {
			binary.LittleEndian.PutUint16(payload[position*2:], token)
			position++
		}
	}
	binary.LittleEndian.PutUint64(offsets[len(records)*8:], uint64(position))
	blob := append(append(append(header, offsets...), labelBytes...), payload...)
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeNewBobDatasetManifest(t *testing.T, dir string, records [][]uint16, labels []int32) {
	t.Helper()
	classCounts := map[string]int64{"0": 0, "1": 0}
	tokens := int64(0)
	for i, record := range records {
		tokens += int64(len(record))
		if labels[i] == 0 {
			classCounts["0"]++
		} else {
			classCounts["1"]++
		}
	}
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens,
		Modality:       "text", VocabSize: 8, TokenDType: data.DatasetTokenDTypeUint16,
		ShardFormat:    data.DatasetShardFormatLabeledSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow, RecordSeqLen: 4,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
		Task:            &data.DatasetTask{Type: data.DatasetTaskSingleLabelClassification, NumLabels: 2},
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: tokens, Shards: 1, Sequences: int64(len(records)), MaxSequenceTokens: 2, ClassCounts: classCounts},
			"val":   {Pattern: "val_*.bin", Tokens: tokens, Shards: 1, Sequences: int64(len(records)), MaxSequenceTokens: 2, ClassCounts: classCounts},
		},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}
