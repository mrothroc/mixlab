package train

import (
	"fmt"
	"os"
	"path/filepath"
)

// checkpointPath returns the safetensors path for a given training step under
// the configured checkpoint directory.
func checkpointPath(dir string, step int) string {
	return filepath.Join(dir, fmt.Sprintf("step_%06d.st", step))
}

// writeCheckpoint reads the live trainer weights and writes a safetensors
// checkpoint for the given step. The directory is created if missing.
func writeCheckpoint(cfg *ArchConfig, trainer GPUTrainer, shapes []WeightShape, dir string, step int) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create checkpoint dir %q: %w", dir, err)
	}
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		return fmt.Errorf("read trainer weights: %w", err)
	}
	path := checkpointPath(dir, step)
	if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
		return fmt.Errorf("export safetensors %q: %w", path, err)
	}
	return nil
}
