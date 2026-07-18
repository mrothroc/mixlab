package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/data"
)

// logValidatedDatasetManifestForConfig validates the dataset manifest against
// the model config and prints a one-line summary when a manifest is present.
func logValidatedDatasetManifestForConfig(cfg *ArchConfig, shardPattern, name string) error {
	manifest, path, err := validateDatasetManifestForConfig(cfg, shardPattern)
	if err != nil {
		return err
	}
	if manifest != nil {
		fmt.Printf("  [%s] dataset manifest: modality=%s representation=%s vocab_size=%d (%s)\n",
			name, manifest.Modality, manifest.Representation, manifest.VocabSize, path)
	}
	return nil
}

func validateDatasetManifestForConfig(cfg *ArchConfig, shardPattern string) (*data.DatasetManifest, string, error) {
	if cfg == nil {
		return nil, "", fmt.Errorf("nil model config")
	}
	manifest, path, found, err := data.FindDatasetManifest(shardPattern)
	if err != nil {
		return nil, "", err
	}
	if !found {
		return nil, "", nil
	}
	if err := manifest.ValidateModelVocab(cfg.VocabSize); err != nil {
		return nil, "", fmt.Errorf("dataset manifest %q is incompatible with config %q: %w", path, cfg.Name, err)
	}
	return manifest, path, nil
}
