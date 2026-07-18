package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestValidateDatasetManifestForConfigOptionalAndCompatible(t *testing.T) {
	dir := t.TempDir()
	pattern := filepath.Join(dir, "train_*.bin")
	if manifest, path, err := validateDatasetManifestForConfig(objectiveTestConfig(), pattern); err != nil || manifest != nil || path != "" {
		t.Fatalf("legacy manifest lookup manifest=%+v path=%q err=%v", manifest, path, err)
	}

	cfg := objectiveTestConfig()
	writeTrainingDatasetManifest(t, dir, cfg.VocabSize)
	manifest, path, err := validateDatasetManifestForConfig(cfg, pattern)
	if err != nil {
		t.Fatal(err)
	}
	if manifest == nil || manifest.Modality != "text" || path != filepath.Join(dir, data.DatasetManifestFilename) {
		t.Fatalf("manifest=%+v path=%q", manifest, path)
	}
}

func TestValidateDatasetManifestForConfigRejectsVocabMismatch(t *testing.T) {
	dir := t.TempDir()
	writeTrainingDatasetManifest(t, dir, 31)
	_, _, err := validateDatasetManifestForConfig(objectiveTestConfig(), filepath.Join(dir, "train_*.bin"))
	if err == nil || !strings.Contains(err.Error(), "dataset vocab_size=31 does not match model vocab_size=32") {
		t.Fatalf("error=%v", err)
	}
}

func TestRunTrainRejectsDatasetManifestBeforeTrainerConstruction(t *testing.T) {
	dir := t.TempDir()
	writeTrainingDatasetManifest(t, dir, 31)
	cfg := objectiveTestConfig()
	_, err := runTrain(cfg, filepath.Join(dir, "train_*.bin"), TrainOptions{})
	if err == nil || !strings.Contains(err.Error(), "dataset vocab_size=31 does not match model vocab_size=32") {
		t.Fatalf("error=%v", err)
	}
}

func writeTrainingDatasetManifest(t *testing.T, dir string, vocabSize int) {
	t.Helper()
	manifest := data.DatasetManifest{
		Format:         data.DatasetManifestFormat,
		Version:        data.DatasetManifestVersion,
		Representation: data.DatasetRepresentationDiscreteTokens,
		Modality:       "text",
		VocabSize:      vocabSize,
		TokenDType:     data.DatasetTokenDTypeUint16,
		ShardFormat:    data.DatasetShardFormatTokenStreamV1,
		Splits: map[string]data.DatasetSplit{
			"train": {Pattern: "train_*.bin", Tokens: 10, Shards: 1},
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
