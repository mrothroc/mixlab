package train

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/gridpoints"
)

func trainingPrecision(cfg *ArchConfig) (gpu.PrecisionPolicy, error) {
	return gpu.ConfigurePrecision(cfg.Training.EffectiveComputeDType(), cfg.Training.AllowTF32)
}

func startTrainingPrecision(cfg *ArchConfig, opts TrainOptions) error {
	p, err := trainingPrecision(cfg)
	if err != nil {
		return err
	}
	// Do not overwrite an existing run's provenance before accepting its resume.
	if opts.Resume != "" {
		var saved *gpu.PrecisionPolicy
		var configHash string
		if cfg.Training.Distributed != nil {
			m, err := resolveDistributedResumeManifest(opts.Resume)
			if err != nil {
				return err
			}
			saved = m.Precision
			configHash = m.ConfigHash
		} else {
			m, err := resolveResumeManifest(opts.Resume)
			if err != nil {
				return err
			}
			saved = m.Precision
			configHash = m.ConfigHash
		}
		if err := checkResumePrecision(saved, p); err != nil {
			return err
		}
		if cfg.GridEnabled() {
			if err := checkResumeConfigHash(cfg, configHash); err != nil {
				return err
			}
		}
	}
	fmt.Printf("  [%s] %s\n", cfg.Name, p)
	writeGridConvolutionContext(os.Stderr, cfg, p)
	// No new output flag: attach provenance to the existing artifact location.
	path := ""
	if opts.SafetensorsPath != "" {
		path = opts.SafetensorsPath + ".run.json"
	} else if opts.CheckpointDir != "" {
		path = filepath.Join(opts.CheckpointDir, "run.json")
	}
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	var points gridpoints.Columns
	if cfg.GridEnabled() {
		points = cfg.InputAdapter.VectorPoints
	}
	return atomicWriteJSON(path, struct {
		Format       string              `json:"format"`
		Name         string              `json:"name"`
		Precision    gpu.PrecisionPolicy `json:"precision"`
		VectorPoints gridpoints.Columns  `json:"vector_points,omitempty"`
	}{"mixlab_run_v1", cfg.Name, p, points})
}

func checkResumePrecision(saved *gpu.PrecisionPolicy, current gpu.PrecisionPolicy) error {
	if saved == nil {
		// Portable checkpoint unit tests have no numerical backend to preserve.
		if current.Backend == "unavailable" {
			return nil
		}
		return fmt.Errorf("checkpoint precision is unknown (legacy manifest); exact resume is unsafe; use a weights-only warm start for a new run")
	}
	if *saved != current {
		return fmt.Errorf("checkpoint precision mismatch: saved=%+v current=%+v; use a fresh weights-only run to change precision", *saved, current)
	}
	return nil
}
