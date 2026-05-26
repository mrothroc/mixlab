package train

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCharFeaturesValidatesHeaderAndPayload(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "char_features.bin")
	vocabSize := 3
	charVocabSize := 257
	charMaxPerToken := 4
	rows := []uint16{
		0, 0, 0, 0,
		66, 67, 0, 0,
		1, 2, 3, 4,
	}
	writeTestCharFeatures(t, path, vocabSize, charVocabSize, charMaxPerToken, rows)

	got, err := loadCharFeatures(path, vocabSize, charVocabSize, charMaxPerToken)
	if err != nil {
		t.Fatalf("loadCharFeatures: %v", err)
	}
	want := []int32{
		0, 0, 0, 0,
		66, 67, 0, 0,
		1, 2, 3, 4,
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("id[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func TestConfigureCharFeaturesForTrainingFindsShardAdjacentFile(t *testing.T) {
	dir := t.TempDir()
	shardPath := filepath.Join(dir, "train_00000.bin")
	if err := os.WriteFile(shardPath, []byte("placeholder"), 0o644); err != nil {
		t.Fatalf("write shard placeholder: %v", err)
	}
	featuresPath := filepath.Join(dir, "char_features.bin")
	writeTestCharFeatures(t, featuresPath, 2, 257, 2, []uint16{
		0, 0,
		42, 43,
	})

	cfg := &ArchConfig{
		VocabSize:       2,
		CharVocabSize:   257,
		CharDim:         8,
		CharMaxPerToken: 2,
	}
	source, err := configureCharFeaturesForTraining(cfg, filepath.Join(dir, "train_*.bin"))
	if err != nil {
		t.Fatalf("configureCharFeaturesForTraining: %v", err)
	}
	if source != featuresPath {
		t.Fatalf("source=%q want %q", source, featuresPath)
	}
	if len(cfg.CharFeatureIDs) != 4 || cfg.CharFeatureIDs[2] != 42 || cfg.CharFeatureIDs[3] != 43 {
		t.Fatalf("unexpected char feature ids: %v", cfg.CharFeatureIDs)
	}
}

func writeTestCharFeatures(t *testing.T, path string, vocabSize, charVocabSize, charMaxPerToken int, rows []uint16) {
	t.Helper()
	if len(rows) != vocabSize*charMaxPerToken {
		t.Fatalf("rows len=%d want %d", len(rows), vocabSize*charMaxPerToken)
	}
	header := make([]int32, charFeatureHeaderInts)
	header[0] = charFeatureMagic
	header[1] = charFeatureVersion
	header[2] = int32(vocabSize)
	header[3] = int32(charVocabSize)
	header[4] = int32(charMaxPerToken)
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create features: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, header); err != nil {
		_ = f.Close()
		t.Fatalf("write header: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, rows); err != nil {
		_ = f.Close()
		t.Fatalf("write rows: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close features: %v", err)
	}
}
