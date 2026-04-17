package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

type optimizerClass int

const (
	optimizerClassEmbed optimizerClass = iota
	optimizerClassHead
	optimizerClassScalar
	optimizerClassMatrix
)

func classifyWeightOptimizer(ws WeightShape) (optimizerClass, error) {
	switch {
	case ws.Name == "embed" || ws.Name == "bigram_table":
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

//nolint:unused // Used by the MLX trainer when built with -tags mlx.
func buildOptimizerGroups(cfg *ArchConfig, shapes []WeightShape) ([]gpu.OptimizerGroup, []gpu.WeightOptimizer, error) {
	groupIndexByClass := make(map[optimizerClass]int, 4)
	groups := make([]gpu.OptimizerGroup, 0, 4)
	weights := make([]gpu.WeightOptimizer, 0, len(shapes))
	addGroup := func(class optimizerClass, group gpu.OptimizerGroup) int {
		if idx, ok := groupIndexByClass[class]; ok {
			return idx
		}
		idx := len(groups)
		groups = append(groups, group)
		groupIndexByClass[class] = idx
		return idx
	}

	muonNesterov := true
	if cfg.Training.MuonNesterov != nil {
		muonNesterov = *cfg.Training.MuonNesterov
	}
	for _, ws := range shapes {
		class, err := classifyWeightOptimizer(ws)
		if err != nil {
			return nil, nil, err
		}
		var group gpu.OptimizerGroup
		switch class {
		case optimizerClassEmbed:
			group = gpu.OptimizerGroup{
				Kind:        gpu.OptimizerAdamW,
				LR:          cfg.Training.EmbedLR,
				Beta1:       cfg.Training.Beta1,
				Beta2:       cfg.Training.Beta2,
				Epsilon:     cfg.Training.Epsilon,
				WeightDecay: cfg.Training.EmbedWeightDecay,
			}
		case optimizerClassHead:
			group = gpu.OptimizerGroup{
				Kind:        gpu.OptimizerAdamW,
				LR:          cfg.Training.HeadLR,
				Beta1:       cfg.Training.Beta1,
				Beta2:       cfg.Training.Beta2,
				Epsilon:     cfg.Training.Epsilon,
				WeightDecay: cfg.Training.HeadWeightDecay,
			}
		case optimizerClassScalar:
			group = gpu.OptimizerGroup{
				Kind:        gpu.OptimizerAdamW,
				LR:          cfg.Training.ScalarLR,
				Beta1:       cfg.Training.Beta1,
				Beta2:       cfg.Training.Beta2,
				Epsilon:     cfg.Training.Epsilon,
				WeightDecay: cfg.Training.ScalarWeightDecay,
			}
		case optimizerClassMatrix:
			group = gpu.OptimizerGroup{
				Kind:         gpu.OptimizerMuon,
				LR:           cfg.Training.MatrixLR,
				Beta1:        cfg.Training.MuonMomentum,
				Beta2:        cfg.Training.Beta2,
				Epsilon:      cfg.Training.Epsilon,
				WeightDecay:  cfg.Training.MatrixWeightDecay,
				BackendSteps: cfg.Training.MuonBackendSteps,
				Nesterov:     muonNesterov,
			}
		default:
			return nil, nil, fmt.Errorf("unsupported optimizer class for %q", ws.Name)
		}
		groupIdx := addGroup(class, group)
		weights = append(weights, gpu.WeightOptimizer{
			GroupIndex: groupIdx,
			Decay:      shouldDecayWeight(ws.Shape),
		})
	}
	return groups, weights, nil
}

func isScalarOptimizerName(name string) bool {
	switch name {
	case "bigram_scale", "decay", "scan_decay", "w_decay", "mu", "mu2":
		return true
	}
	return strings.HasSuffix(name, "_scale")
}
