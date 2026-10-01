package train

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
)

// Best-file selection is independent of replayed metric state: validation may
// have published a better artifact after the checkpoint we are resuming from.
type gridBestArtifact struct {
	path, configHash, datasetHash string
	score                         float64
}

func newGridBestArtifact(cfg *ArchConfig, dir, datasetHash string, resume bool) (*gridBestArtifact, error) {
	if dir == "" {
		return nil, nil
	}
	hash, err := resumeConfigHash(cfg)
	if err != nil {
		return nil, err
	}
	b := &gridBestArtifact{path: filepath.Join(dir, "best.safetensors"), configHash: hash, datasetHash: datasetHash, score: math.Inf(1)}
	if !resume {
		return b, nil
	}
	var meta map[string]string
	if _, err = loadSafetensorsWithMetadata(b.path, &meta); errors.Is(err, os.ErrNotExist) {
		return b, nil
	} else if err != nil {
		return nil, fmt.Errorf("read grid best checkpoint: %w", err)
	}
	if meta["grid_best_config_hash"] != hash || meta["grid_best_dataset_hash"] != datasetHash {
		return nil, fmt.Errorf("grid best checkpoint lacks matching config/dataset selection metadata; use a fresh checkpoint directory")
	}
	b.score, err = strconv.ParseFloat(meta["grid_best_masked_rmse"], 64)
	if err != nil || b.score < 0 || math.IsNaN(b.score) || math.IsInf(b.score, 0) {
		return nil, fmt.Errorf("grid best checkpoint has invalid selection score")
	}
	return b, nil
}

func (b *gridBestArtifact) save(cfg *ArchConfig, trainer any, shapes []WeightShape, score float64) error {
	if b == nil || score >= b.score {
		return nil
	}
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(b.path), 0755); err != nil {
		return err
	}
	// Weights and their score share one atomic safetensors replacement.
	err = exportSafetensorsWithMetadata(b.path, cfg, shapes, weights, map[string]string{
		"grid_best_config_hash": b.configHash, "grid_best_dataset_hash": b.datasetHash,
		"grid_best_masked_rmse": strconv.FormatFloat(score, 'g', -1, 64),
	})
	if err == nil {
		b.score = score
	}
	return err
}
