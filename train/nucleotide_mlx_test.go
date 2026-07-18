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

func TestNucleotideTinyCausalAndMLMTrainingSmoke(t *testing.T) {
	for _, objective := range []string{"causal", "mlm"} {
		t.Run(objective, func(t *testing.T) {
			dir := t.TempDir()
			writeNucleotideMLXDataset(t, dir)
			mlmFields := ""
			if objective == "mlm" {
				mlmFields = `,"mlm_head":"bert"`
			}
			config := []byte(`{
                "name":"nucleotide-` + objective + `-mlx",
                "model_dim":16,"vocab_size":9,"seq_len":8,"mlp_mult":2,"tie_embeddings":true` + mlmFields + `,
                "blocks":[{"type":"plain","heads":2},{"type":"swiglu"}],
                "training":{"objective":"` + objective + `","steps":12,"lr":0.001,"batch_tokens":16,"seed":7,
                  "mlm_mask_prob":0.5,"mlm_mask_token_id":3,"reverse_complement_prob":0.5,
                  "weight_decay":0,"grad_clip":1}
              }`)
			cfg, err := ParseArchConfig(config, "nucleotide-smoke")
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
		})
	}
}

func writeNucleotideMLXDataset(t *testing.T, dir string) {
	t.Helper()
	vocab := data.NucleotideVocabulary{
		Format: data.NucleotideVocabularyFormat, Version: data.NucleotideVocabularyVersion,
		Alphabet: data.NucleotideAlphabetDNA, InvalidSymbolPolicy: "error", AmbiguousSymbols: []string{"N"},
		Tokens:      map[string]int{"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<MASK>": 3, "A": 4, "C": 5, "G": 6, "T": 7, "N": 8},
		Complements: map[string]string{"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"},
	}
	blob, _ := json.Marshal(vocab)
	if err := os.WriteFile(filepath.Join(dir, "nucleotide_vocab.json"), blob, 0o644); err != nil {
		t.Fatal(err)
	}
	records := [][]uint16{
		{4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7},
		{7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4},
		{4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 5, 5},
		{7, 7, 6, 6, 5, 5, 4, 4, 7, 7, 6, 6},
	}
	writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "train_00000.bin"), records)
	writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "val_00000.bin"), records)
	manifest := data.DatasetManifest{
		Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens, Modality: "nucleotide", VocabSize: 9,
		TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatSequenceV1,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2, "mask": 3},
		Artifacts:       data.DatasetManifestArtifacts{Vocabulary: "nucleotide_vocab.json"},
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: 48, Shards: 1, Sequences: 4},
			"val":   {Pattern: "val_*.bin", Tokens: 48, Shards: 1, Sequences: 4},
		},
	}
	blob, _ = json.Marshal(manifest)
	if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeNucleotideMLXSequenceShard(t *testing.T, path string, records [][]uint16) {
	t.Helper()
	const headerInts = 256
	header := make([]byte, headerInts*4)
	total := 0
	for _, record := range records {
		total += len(record)
	}
	binary.LittleEndian.PutUint32(header[0:4], 20260718)
	binary.LittleEndian.PutUint32(header[4:8], 1)
	binary.LittleEndian.PutUint32(header[8:12], uint32(total))
	binary.LittleEndian.PutUint32(header[12:16], uint32(len(records)))
	offsets := make([]byte, (len(records)+1)*8)
	payload := make([]byte, total*2)
	offset := 0
	for i, record := range records {
		binary.LittleEndian.PutUint64(offsets[i*8:], uint64(offset))
		for _, token := range record {
			binary.LittleEndian.PutUint16(payload[offset*2:], token)
			offset++
		}
	}
	binary.LittleEndian.PutUint64(offsets[len(records)*8:], uint64(offset))
	if err := os.WriteFile(path, append(append(header, offsets...), payload...), 0o644); err != nil {
		t.Fatal(err)
	}
}
