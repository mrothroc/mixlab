package gpu

import (
	"fmt"
	"strings"
)

type OptimizerWeightMetadata struct {
	Name         string
	Shape        []int
	IsBuffer     bool
	IsNormScale  bool
	Group        string
	ForceNoDecay bool
	ForceDecay   bool
}

type OptimizerSettings struct {
	Name                              string
	LR                                float32
	Beta1                             float32
	Beta2                             float32
	Epsilon                           float32
	WeightDecay                       float32
	LAMBTrustRatioCap                 float32
	CautiousWeightDecay               bool
	CautiousWeightDecayActivationStep int
	BackendSteps                      int
	NewtonSchulzVariant               string
	Nesterov                          bool
	MuonNormalization                 MuonNormalization
	// RowNormalize is retained for direct TrainerOptimizerSpec callers.
	// Config-driven code should prefer MuonNormalization.
	RowNormalize bool
}

type TrainerOptimizerConfig struct {
	Weights       []OptimizerWeightMetadata
	Embed         OptimizerSettings
	Head          OptimizerSettings
	Scalar        OptimizerSettings
	Matrix        OptimizerSettings
	ExtraGroups   map[string]OptimizerSettings
	DecayAll      bool
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
	groupIndexByKey := make(map[string]int, 4+len(cfg.ExtraGroups))
	groups := make([]OptimizerGroup, 0, 4)
	weights := make([]WeightOptimizer, 0, len(cfg.Weights))
	addGroup := func(key string, settings OptimizerSettings) (int, error) {
		if idx, ok := groupIndexByKey[key]; ok {
			return idx, nil
		}
		group, err := optimizerGroup(settings)
		if err != nil {
			return 0, err
		}
		idx := len(groups)
		groups = append(groups, group)
		groupIndexByKey[key] = idx
		return idx, nil
	}

	for _, weight := range cfg.Weights {
		if weight.IsBuffer {
			weights = append(weights, WeightOptimizer{Frozen: true})
			continue
		}
		class, err := classifyWeightOptimizer(weight)
		if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		groupKey := fmt.Sprintf("class:%d", class)
		settings, err := optimizerSettingsForClass(cfg, class, weight.Name)
		if weight.Group != "" {
			var ok bool
			settings, ok = cfg.ExtraGroups[weight.Group]
			if !ok {
				return TrainerOptimizerSpec{}, fmt.Errorf("weight %q references unknown optimizer group %q", weight.Name, weight.Group)
			}
			groupKey = "extra:" + weight.Group
		} else if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		groupIdx, err := addGroup(groupKey, settings)
		if err != nil {
			return TrainerOptimizerSpec{}, err
		}
		weights = append(weights, WeightOptimizer{
			GroupIndex: groupIdx,
			Decay:      shouldDecayOptimizerWeightWithPolicy(weight, class, cfg.DecayAll),
		})
	}

	return TrainerOptimizerSpec{
		Groups:        groups,
		Weights:       weights,
		MaxGradNorm:   cfg.MaxGradNorm,
		DefaultBaseLR: cfg.DefaultBaseLR,
	}, nil
}

func shouldDecayOptimizerWeightWithPolicy(weight OptimizerWeightMetadata, class optimizerClass, decayAll bool) bool {
	if weight.ForceNoDecay {
		return false
	}
	if weight.ForceDecay {
		return true
	}
	if decayAll {
		return true
	}
	return shouldDecayOptimizerWeight(weight.Shape, class)
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
	muonNormalization := settings.MuonNormalization
	if muonNormalization == MuonNormalizationNone {
		switch strings.ToLower(strings.TrimSpace(settings.Name)) {
		case "muon_eq_r":
			muonNormalization = MuonNormalizationRowL2
		case "normuon":
			muonNormalization = MuonNormalizationNorMuon
		default:
			if settings.RowNormalize {
				muonNormalization = MuonNormalizationRowL2
			}
		}
	}
	return OptimizerGroup{
		Kind:                              kind,
		LR:                                settings.LR,
		Beta1:                             settings.Beta1,
		Beta2:                             settings.Beta2,
		Epsilon:                           settings.Epsilon,
		WeightDecay:                       settings.WeightDecay,
		LAMBTrustRatioCap:                 settings.LAMBTrustRatioCap,
		CautiousWeightDecay:               settings.CautiousWeightDecay,
		CautiousWeightDecayActivationStep: settings.CautiousWeightDecayActivationStep,
		BackendSteps:                      settings.BackendSteps,
		NewtonSchulzVariant:               parseNewtonSchulzVariant(settings.NewtonSchulzVariant),
		Nesterov:                          settings.Nesterov,
		MuonNormalization:                 muonNormalization,
		RowNormalize:                      settings.RowNormalize,
	}, nil
}

func optimizerKind(name string) (OptimizerKind, error) {
	switch strings.ToLower(name) {
	case "adamw":
		return OptimizerAdamW, nil
	case "muon", "muon_eq_r", "normuon":
		return OptimizerMuon, nil
	case "sgd":
		return OptimizerSGD, nil
	case "lamb":
		return OptimizerLAMB, nil
	default:
		return 0, fmt.Errorf("unsupported optimizer %q", name)
	}
}

func classifyWeightOptimizer(ws OptimizerWeightMetadata) (optimizerClass, error) {
	switch {
	case ws.Name == "embed" || ws.Name == "rtd_generator_embed" || ws.Name == "char_table" || ws.Name == "bigram_table" || ws.Name == "trigram_table":
		return optimizerClassEmbed, nil
	case ws.Name == "head" || strings.HasPrefix(ws.Name, "head_"):
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

func shouldDecayOptimizerWeight(shape []int, class optimizerClass) bool {
	if class == optimizerClassScalar {
		return false
	}
	return shouldDecayWeight(shape)
}

func isScalarOptimizerName(name string) bool {
	switch name {
	case "bigram_scale", "trigram_scale", "smear_gate", "smear_scale", "backout_lambda", "decay", "scan_decay", "w_decay", "mu", "mu2",
		"s4d_log_dt", "s4d_log_A_real", "s4d_A_imag", "s4d_B_real", "s4d_B_imag",
		"s4d_C_real", "s4d_C_imag", "s4d_C_backward_real", "s4d_C_backward_imag", "s4d_D":
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
