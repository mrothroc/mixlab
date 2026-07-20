//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestPerRecordFramingTinyCausalTrainingSmoke(t *testing.T) {
	dir := t.TempDir()
	records := [][]uint16{
		{3, 4, 5}, {3, 4, 5}, {5, 4, 3}, {5, 4, 3},
		{3, 3, 4, 4}, {5, 5, 4, 4}, {3, 4}, {5, 4},
	}
	writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "train_00000.bin"), records)
	writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "val_00000.bin"), records)
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens, Modality: "text", VocabSize: 6,
		TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatSequenceV1,
		SequenceLayout: data.DatasetSequenceLayoutOneRecordRow, RecordSeqLen: 8,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
		Artifacts:       data.DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: 24, Shards: 1, Sequences: 8, MaxSequenceTokens: 4},
			"val":   {Pattern: "val_*.bin", Tokens: 24, Shards: 1, Sequences: 8, MaxSequenceTokens: 4},
		},
	}
	blob, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig([]byte(`{
      "name":"record-framing-mlx","model_dim":16,"vocab_size":6,"seq_len":8,
      "mlp_mult":2,"tie_embeddings":true,
      "blocks":[{"type":"plain","heads":2},{"type":"swiglu"}],
      "training":{"objective":"causal","steps":16,"lr":0.001,"batch_tokens":16,
        "seed":7,"weight_decay":0,"grad_clip":1}
    }`), "record-framing-smoke")
	if err != nil {
		t.Fatal(err)
	}
	result, err := runTrain(cfg, filepath.Join(dir, "train_*.bin"), TrainOptions{LogEvery: 100, ValEvery: 100})
	if err != nil {
		t.Fatalf("runTrain: %v", err)
	}
	if math.IsNaN(result.FirstLoss) || math.IsInf(result.FirstLoss, 0) || math.IsNaN(result.LastLoss) || math.IsInf(result.LastLoss, 0) {
		t.Fatalf("non-finite loss first=%g last=%g", result.FirstLoss, result.LastLoss)
	}
	if result.LastLoss >= result.FirstLoss {
		t.Fatalf("loss did not decrease: first=%g last=%g", result.FirstLoss, result.LastLoss)
	}
}
