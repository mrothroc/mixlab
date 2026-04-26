package gpu

import (
	"fmt"
	"strings"
)

type OptimizerWeightMetadata struct {
	Name        string
	Shape       []int
	IsNormScale bool
}

type OptimizerSettings struct {
	Name                string
	LR                  float32
	Beta1               float32
	Beta2               float32
	Epsilon             float32
	WeightDecay         float32
	BackendSteps        int
	NewtonSchulzVariant string
	Nesterov            bool
}

type TrainerOptimizerConfig struct {
	Weights       []OptimizerWeightMetadata
	Embed         OptimizerSettings
	Head          OptimizerSettings
	Scalar        OptimizerSettings
	Matrix        OptimizerSettings
	MaxGradNorm   float32
	DefaultBaseLR float32
}

type optimizerClass int

const (
	optimizerClassEmbed optimizerClass = iota
	optimizerClassHead
	optimizerClassScalar
	optimizerClassMatrix
)

func BuildTrainerOptimizerSpec(cfg TrainerOptimizerConfig) (TrainerOptimizerSpec, error) {
	groupIndexByClass := make(map[optimizerClass]int, 4)
	groups := make([]OptimizerGroup, 0, 4)
	weights := make([]WeightOptimizer, 0, len(cfg.Weights))
	addGroup := func(class optimizerClass, settings OptimizerSettings) (int, error) {
		if idx, ok := groupIndexByClass[class]; ok {
			return idx, nil
		}
		group, err := optimizerGroup(settings)
		if err != nil {
			return 0, err
		}
		idx := len(groups)
		groups = append(groups, group)
		groupIndexByClass[class] = idx
		return idx, nil
	}

	for _, weight := range cfg.Weights {
		class, err := classifyWeightOptimizer(weight)
		if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		settings, err := optimizerSettingsForClass(cfg, class, weight.Name)
		if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		groupIdx, err := addGroup(class, settings)
		if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		weights = append(weights, WeightOptimizer{
			GroupIndex: groupIdx,
			Decay:      shouldDecayWeight(weight.Shape),
		})
	}

	return TrainerOptimizerSpec{
		Groups:        groups,
		Weights:       weights,
		MaxGradNorm:   cfg.MaxGradNorm,
		DefaultBaseLR: cfg.DefaultBaseLR,
	}, nil
}

func optimizerSettingsForClass(cfg TrainerOptimizerConfig, class optimizerClass, weightName string) (OptimizerSettings, error) {
	switch class {
	case optimizerClassEmbed:
		return cfg.Embed, nil
	case optimizerClassHead:
		return cfg.Head, nil
	case optimizerClassScalar:
		return cfg.Scalar, nil
	case optimizerClassMatrix:
		return cfg.Matrix, nil
	default:
		return OptimizerSettings{}, fmt.Errorf("unsupported optimizer class for %q", weightName)
	}
}

func optimizerGroup(settings OptimizerSettings) (OptimizerGroup, error) {
	kind, err := optimizerKind(settings.Name)
	if err != nil {
		return OptimizerGroup{}, err
	}
	return OptimizerGroup{
		Kind:                kind,
		LR:                  settings.LR,
		Beta1:               settings.Beta1,
		Beta2:               settings.Beta2,
		Epsilon:             settings.Epsilon,
		WeightDecay:         settings.WeightDecay,
		BackendSteps:        settings.BackendSteps,
		NewtonSchulzVariant: parseNewtonSchulzVariant(settings.NewtonSchulzVariant),
		Nesterov:            settings.Nesterov,
	}, nil
}

func optimizerKind(name string) (OptimizerKind, error) {
	switch strings.ToLower(name) {
	case "adamw":
		return OptimizerAdamW, nil
	case "muon":
		return OptimizerMuon, nil
	default:
		return 0, fmt.Errorf("unsupported optimizer %q", name)
	}
}

func classifyWeightOptimizer(ws OptimizerWeightMetadata) (optimizerClass, error) {
	switch {
	case ws.Name == "embed" || ws.Name == "bigram_table" || ws.Name == "trigram_table":
		return optimizerClassEmbed, nil
	case ws.Name == "head":
		return optimizerClassHead, nil
	case ws.IsNormScale:
		return optimizerClassScalar, nil
	case isScalarOptimizerName(ws.Name):
		return optimizerClassScalar, nil
	case len(ws.Shape) == 1:
		return optimizerClassScalar, nil
	case len(ws.Shape) == 2:
		return optimizerClassMatrix, nil
	default:
		return 0, fmt.Errorf("unclassified weight %q with shape %v", ws.Name, ws.Shape)
	}
}

func shouldDecayWeight(shape []int) bool {
	return len(shape) >= 2
}

func isScalarOptimizerName(name string) bool {
	switch name {
	case "bigram_scale", "trigram_scale", "decay", "scan_decay", "w_decay", "mu", "mu2":
		return true
	}
	return strings.HasSuffix(name, "_scale")
}

func parseNewtonSchulzVariant(name string) NewtonSchulzVariant {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "polar_express":
		return NewtonSchulzPolarExpress
	default:
		return NewtonSchulzFixed
	}
}
