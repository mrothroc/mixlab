package train

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// checkpointPath returns the safetensors path for a given training step under
// the configured checkpoint directory.
func checkpointPath(dir string, step int) string {
	return filepath.Join(dir, fmt.Sprintf("step_%06d.st", step))
}

type safetensorsArtifacts struct {
	FinalPath string
	SWAPath   string
}

func (a safetensorsArtifacts) Summary() string {
	switch {
	case a.FinalPath != "" && a.SWAPath != "":
		return fmt.Sprintf("final=%s swa=%s", a.FinalPath, a.SWAPath)
	case a.FinalPath != "":
		return a.FinalPath
	case a.SWAPath != "":
		return a.SWAPath
	default:
		return "none"
	}
}

func suffixedSafetensorsPaths(path string) safetensorsArtifacts {
	base := path
	for _, ext := range []string{".safetensors", ".st"} {
		if strings.HasSuffix(base, ext) {
			base = strings.TrimSuffix(base, ext)
			break
		}
	}
	return safetensorsArtifacts{
		FinalPath: base + ".final.safetensors",
		SWAPath:   base + ".swa.safetensors",
	}
}

func checkpointSuffixedSafetensorsPaths(dir string, step int) safetensorsArtifacts {
	return suffixedSafetensorsPaths(filepath.Join(dir, fmt.Sprintf("step_%06d.safetensors", step)))
}

// writeCheckpoint reads the live trainer weights and writes a safetensors
// checkpoint for the given step. When SWA/EMA weights are available, it writes
// both live final and averaged checkpoint files. The directory is created if
// missing.
func writeCheckpoint(cfg *ArchConfig, trainer any, shapes []WeightShape, dir string, step int, swaEMA [][]float32) (safetensorsArtifacts, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return safetensorsArtifacts{}, fmt.Errorf("create checkpoint dir %q: %w", dir, err)
	}
	weights, err := readTrainerWeights(trainer)
	if err != nil {
		return safetensorsArtifacts{}, fmt.Errorf("read trainer weights: %w", err)
	}
	if hasSWAWeights(swaEMA) {
		artifacts := checkpointSuffixedSafetensorsPaths(dir, step)
		if err := exportSafetensors(artifacts.FinalPath, cfg, shapes, weights); err != nil {
			return safetensorsArtifacts{}, fmt.Errorf("export final safetensors %q: %w", artifacts.FinalPath, err)
		}
		if err := exportSafetensors(artifacts.SWAPath, cfg, shapes, swaEMA); err != nil {
			return safetensorsArtifacts{}, fmt.Errorf("export swa safetensors %q: %w", artifacts.SWAPath, err)
		}
		return artifacts, nil
	}
	artifacts := safetensorsArtifacts{FinalPath: checkpointPath(dir, step)}
	if err := exportSafetensors(artifacts.FinalPath, cfg, shapes, weights); err != nil {
		return safetensorsArtifacts{}, fmt.Errorf("export safetensors %q: %w", artifacts.FinalPath, err)
	}
	return artifacts, nil
}
