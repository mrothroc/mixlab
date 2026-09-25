package data

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestDistributedRecordPartitionAndRestore(t *testing.T) {
	dir := t.TempDir()
	records := [][]uint16{{4}, {5, 5}, {6, 6, 6}, {7, 7, 7, 7}}
	writeSequenceShardFixture(t, filepath.Join(dir, "train.bin"), records)
	m := DatasetManifest{Format: DatasetManifestFormat, Version: DatasetManifestVersion,
		Representation: DatasetRepresentationDiscreteTokens, Modality: "text", VocabSize: 16,
		TokenDType: DatasetTokenDTypeUint16, ShardFormat: DatasetShardFormatSequenceV1,
		SequenceLayout: DatasetSequenceLayoutOneRecordRow, RecordSeqLen: 6,
		SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
		Artifacts:       DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
		Splits:          map[string]DatasetSplit{"train": {Pattern: "train.bin", Tokens: 10, Shards: 1, Sequences: 4, MaxSequenceTokens: 4}}}
	blob, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, DatasetManifestFilename), blob, 0600); err != nil {
		t.Fatal(err)
	}
	newLoader := func(rank int) *DistributedLoader {
		l, err := NewDistributedLoader(filepath.Join(dir, "train.bin"), 7, 2, rank, 6, 6, 16, "members")
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	seen := map[int]bool{}
	for rank := 0; rank < 2; rank++ {
		l := newLoader(rank)
		b, err := l.NextBatch(12)
		if err != nil {
			t.Fatal(err)
		}
		for row := 0; row < 2; row++ {
			base := row * 6
			token := b.X[base+1]
			if seen[token] {
				t.Fatal("record owned by multiple ranks")
			}
			seen[token] = true
			length := token - 3
			for j := 0; j < 6; j++ {
				want := float32(0)
				if j <= length {
					want = 1
					if b.Y[base+j] != b.X[base+j+1] {
						t.Fatal("target crosses record")
					}
				}
				if b.LossMask[base+j] != want {
					t.Fatal("incorrect valid-token count")
				}
			}
		}
		restored := newLoader(rank)
		if err := restored.Restore(l.State()); err != nil {
			t.Fatal(err)
		}
		a, err := l.NextBatch(6)
		if err != nil {
			t.Fatal(err)
		}
		b, err = restored.NextBatch(6)
		if err != nil || !reflect.DeepEqual(a, b) {
			t.Fatalf("record resume mismatch: %v", err)
		}
	}
	if len(seen) != 4 {
		t.Fatal("incomplete record coverage")
	}
}

func TestDistributedContentIdentityAndDirectRestore(t *testing.T) {
	dirs := []string{t.TempDir(), t.TempDir()}
	tokens := make([]uint16, 129)
	for i := range tokens {
		tokens[i] = uint16(i)
	}
	for _, dir := range dirs {
		writeShard(t, dir, "train_00.bin", tokens)
	}
	if err := os.Chtimes(filepath.Join(dirs[1], "train_00.bin"), time.Unix(1234, 0), time.Unix(1234, 0)); err != nil {
		t.Fatal(err)
	}
	newLoader := func(dir string, rank int) *DistributedLoader {
		l, e := NewDistributedLoader(filepath.Join(dir, "*.bin"), 47, 2, rank, 4, 8, 256, "members")
		if e != nil {
			t.Fatal(e)
		}
		return l
	}
	a, b := newLoader(dirs[0], 0), newLoader(dirs[1], 1)
	if a.State().DatasetID != b.State().DatasetID {
		t.Fatal("path/mtime changed dataset identity")
	}
	seen := map[int]bool{}
	for i := 0; i < 8; i++ {
		for _, l := range []*DistributedLoader{a, b} {
			batch, e := l.NextBatch(8)
			if e != nil {
				t.Fatal(e)
			}
			for j, x := range batch.X {
				if seen[x] {
					t.Fatalf("overlapping ownership of token %d", x)
				}
				seen[x] = true
				if batch.Y[j] != x+1 {
					t.Fatal("bad causal target")
				}
			}
		}
	}
	if len(seen) != 128 {
		t.Fatalf("coverage=%d", len(seen))
	}
	// Restore midway through a multi-row chunk, on a copied local dataset.
	if _, e := a.NextBatch(4); e != nil {
		t.Fatal(e)
	}
	s := a.State()
	restored := newLoader(dirs[1], 0)
	if e := restored.Restore(s); e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 30; i++ {
		x, e := a.NextBatch(12)
		if e != nil {
			t.Fatal(e)
		}
		y, e := restored.NextBatch(12)
		if e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(x, y) {
			t.Fatal("direct resume differs")
		}
	}
	s.PartitionID = "other"
	if restored.Restore(s) == nil {
		t.Fatal("accepted different partition")
	}
	tokens[3]++
	writeShard(t, dirs[1], "train_00.bin", tokens)
	if newLoader(dirs[1], 0).State().DatasetID == a.State().DatasetID {
		t.Fatal("changed content retained identity")
	}
}

func TestDistributedLoaderRaggedAndBounds(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, dir, "train_00.bin", []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	for rank := 0; rank < 2; rank++ {
		l, e := NewDistributedLoader(filepath.Join(dir, "*.bin"), 1, 2, rank, 4, 4, 11, "m")
		if e != nil {
			t.Fatal(e)
		}
		for i := 0; i < 4; i++ {
			b, e := l.NextBatch(4)
			if e != nil {
				t.Fatal(e)
			}
			for _, x := range b.X {
				if x >= 8 {
					t.Fatal("consumed ragged tail")
				}
			}
		}
	}
	if _, e := NewDistributedLoader(filepath.Join(dir, "*.bin"), 1, 2, 0, 4, 4, 10, "m"); e == nil {
		t.Fatal("accepted out-of-vocab token")
	}
	if _, e := NewDistributedLoader(filepath.Join(dir, "*.bin"), 1, 4, 0, 4, 4, 11, "m"); e == nil {
		t.Fatal("accepted insufficient chunks")
	}
}
