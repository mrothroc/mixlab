package train

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"sort"

	"github.com/mrothroc/mixlab/arch"
)

type gridLoadReport struct {
	Loaded []string
	New    []string
}

func gridWarmStartPath(cfg *ArchConfig, opts TrainOptions) (string, error) {
	path := cfg.Training.InitFrom
	if path != "" && opts.SafetensorsLoad != "" {
		return "", fmt.Errorf("training.init_from and -safetensors-load are mutually exclusive")
	}
	if path != "" && !filepath.IsAbs(path) {
		if cfg.SourcePath == "" {
			return "", fmt.Errorf("relative training.init_from requires a loaded config file (or an absolute path)")
		}
		path = filepath.Join(filepath.Dir(cfg.SourcePath), path)
	}
	if opts.SafetensorsLoad != "" {
		path = opts.SafetensorsLoad
	}
	if opts.Resume != "" && (path != "" || len(cfg.Training.InitAllowMissing) > 0) {
		return "", fmt.Errorf("-resume cannot be combined with warm start or init_allow_missing; a stage transition is a warm start")
	}
	if path == "" && len(cfg.Training.InitAllowMissing) > 0 {
		return "", fmt.Errorf("init_allow_missing requires init_from or -safetensors-load")
	}
	return path, nil
}

// loadGridWeights uses declared logical identities only. Legacy checkpoints
// without metadata must match the exact physical index/name inventory.
func loadGridWeights(path string, cfg *ArchConfig, shapes []WeightShape, allowMissing []string) ([][]float32, gridLoadReport, error) {
	report := gridLoadReport{}
	var metadata map[string]string
	tensors, err := loadSafetensorsWithMetadata(path, &metadata)
	if err != nil {
		return nil, report, err
	}
	names := make([]string, len(shapes))
	for j, s := range shapes {
		names[j] = s.Name
	}
	allowed, err := arch.MatchGridWeightPatterns(allowMissing, names)
	if err != nil {
		return nil, report, err
	}
	identities := map[string]string{}
	if encoded, ok := metadata["logical_weights"]; ok {
		if metadata["task"] != arch.ObjectiveDenseRegression {
			return nil, report, fmt.Errorf("logical weight metadata requires dense_regression task")
		}
		if err = json.Unmarshal([]byte(encoded), &identities); err != nil || identities == nil {
			return nil, report, fmt.Errorf("invalid logical_weights metadata")
		}
	} else {
		if len(allowMissing) > 0 {
			return nil, report, fmt.Errorf("legacy checkpoint requires exact index/name loading; allow-missing needs logical_weights metadata")
		}
		for j, s := range shapes {
			identities[s.Name] = fmt.Sprintf("w%d_%s", j, s.Name)
		}
	}
	wanted, used := map[string]bool{}, map[string]bool{}
	for _, name := range names {
		wanted[name] = true
	}
	var unexpected []string
	for logical, physical := range identities {
		if !wanted[logical] {
			unexpected = append(unexpected, logical)
		}
		if physical == "" || used[physical] {
			return nil, report, fmt.Errorf("ambiguous logical weight mapping for %q", physical)
		}
		if _, ok := tensors[physical]; !ok {
			return nil, report, fmt.Errorf("logical weight %q references missing tensor %q", logical, physical)
		}
		used[physical] = true
	}
	for name := range tensors {
		if !used[name] {
			unexpected = append(unexpected, name)
		}
	}
	if len(unexpected) > 0 {
		sort.Strings(unexpected)
		return nil, report, fmt.Errorf("unexpected checkpoint tensors: %v", unexpected)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	var missing []string
	for j, shape := range shapes {
		physical, ok := identities[shape.Name]
		if !ok {
			if allowed[shape.Name] {
				report.New = append(report.New, shape.Name)
			} else {
				missing = append(missing, shape.Name)
			}
			continue
		}
		values, err := decodeSafetensorFloat32(physical, shape.Shape, tensors)
		if err != nil {
			return nil, report, fmt.Errorf("logical weight %q: %w", shape.Name, err)
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, report, fmt.Errorf("logical weight %q contains non-finite values", shape.Name)
			}
		}
		weights[j] = values
		report.Loaded = append(report.Loaded, shape.Name)
	}
	if len(missing) > 0 {
		return nil, report, fmt.Errorf("missing logical weights (not allowed by init_allow_missing): %v", missing)
	}
	return weights, report, nil
}

func gridTrainableNames(shapes []WeightShape) []string {
	var names []string
	for _, s := range shapes {
		if !s.Frozen {
			names = append(names, s.Name)
		}
	}
	return names
}
