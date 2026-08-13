package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	S4DDiscretizationZOH         = "zoh"
	S4DDiscretizationBilinear    = "bilinear"
	DefaultS4DSobolevLR          = 0.01
	S4DSobolevGranularityChannel = "channel"
	S4DSobolevGranularityLayer   = "layer"

	WeightDecayPolicyMatrixOnly = "matrix_only"
	WeightDecayPolicyAll        = "all"
)

func effectiveS4DFreqScale(spec BlockSpec) float64 {
	if spec.FreqScale == nil {
		return 1
	}
	return *spec.FreqScale
}

// S4DSobolevFilterEnabled reports whether this block has the optional learned
// transfer-function exponent.
func (spec BlockSpec) S4DSobolevFilterEnabled() bool {
	return spec.SobolevFilter != nil && spec.SobolevFilter.Enabled
}

func effectiveS4DSobolevLearningRate(spec BlockSpec) float64 {
	if !spec.S4DSobolevFilterEnabled() || spec.SobolevFilter.LearningRate == nil {
		return DefaultS4DSobolevLR
	}
	return *spec.SobolevFilter.LearningRate
}

func EffectiveS4DSobolevTrainable(spec BlockSpec) bool {
	return spec.S4DSobolevFilterEnabled() &&
		(spec.SobolevFilter.Trainable == nil || *spec.SobolevFilter.Trainable)
}

func EffectiveS4DSobolevWeightDecay(spec BlockSpec) float64 {
	if !spec.S4DSobolevFilterEnabled() || spec.SobolevFilter.WeightDecay == nil {
		return 0
	}
	return *spec.SobolevFilter.WeightDecay
}

func EffectiveS4DSobolevGranularity(spec BlockSpec) string {
	if !spec.S4DSobolevFilterEnabled() {
		return S4DSobolevGranularityChannel
	}
	switch strings.ToLower(strings.TrimSpace(spec.SobolevFilter.Granularity)) {
	case "", S4DSobolevGranularityChannel:
		return S4DSobolevGranularityChannel
	case S4DSobolevGranularityLayer:
		return S4DSobolevGranularityLayer
	default:
		return strings.ToLower(strings.TrimSpace(spec.SobolevFilter.Granularity))
	}
}

func S4DSobolevBounds(spec BlockSpec) (float64, float64, bool) {
	if !spec.S4DSobolevFilterEnabled() || len(spec.SobolevFilter.Bounds) == 0 {
		return 0, 0, false
	}
	if len(spec.SobolevFilter.Bounds) != 2 {
		return 0, 0, false
	}
	return spec.SobolevFilter.Bounds[0], spec.SobolevFilter.Bounds[1], true
}

func EffectiveS4DSobolevBeta(raw float64, spec BlockSpec) float64 {
	lo, hi, bounded := S4DSobolevBounds(spec)
	if !bounded {
		return raw
	}
	return (lo+hi)/2 + (hi-lo)/2*math.Tanh(raw)
}

func s4dSobolevRawInit(spec BlockSpec) float64 {
	lo, hi, bounded := S4DSobolevBounds(spec)
	if !bounded {
		return spec.SobolevFilter.BetaInit
	}
	ratio := (spec.SobolevFilter.BetaInit - (lo+hi)/2) / ((hi - lo) / 2)
	return math.Atanh(ratio)
}

// S4DSobolevWeightBinding identifies one stored beta tensor and every block
// that reuses it through recurrence or a weight group.
type S4DSobolevWeightBinding struct {
	WeightIndex  int
	BlockIndexes []int
	Spec         BlockSpec
}

func CollectS4DSobolevWeightBindings(cfg *ArchConfig) ([]S4DSobolevWeightBinding, error) {
	if cfg == nil {
		return nil, nil
	}
	hasSobolev := false
	for _, block := range cfg.Blocks {
		if block.S4DSobolevFilterEnabled() {
			hasSobolev = true
			break
		}
	}
	if !hasSobolev {
		return nil, nil
	}
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return nil, err
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		return nil, err
	}
	var weightIndexes []int
	for i, meta := range metas {
		if meta.Name == "s4d_sobolev_beta" {
			weightIndexes = append(weightIndexes, i)
		}
	}
	var roots []int
	for i, block := range cfg.Blocks {
		if refs[i] == i && block.S4DSobolevFilterEnabled() {
			roots = append(roots, i)
		}
	}
	if len(roots) != len(weightIndexes) {
		return nil, fmt.Errorf("S4D Sobolev weight binding mismatch: blocks=%d weights=%d", len(roots), len(weightIndexes))
	}
	bindings := make([]S4DSobolevWeightBinding, len(roots))
	for bindingIndex, root := range roots {
		binding := S4DSobolevWeightBinding{WeightIndex: weightIndexes[bindingIndex], Spec: cfg.Blocks[root]}
		for blockIndex, ref := range refs {
			if ref == root {
				binding.BlockIndexes = append(binding.BlockIndexes, blockIndex)
			}
		}
		bindings[bindingIndex] = binding
	}
	return bindings, nil
}

func effectiveS4DNSSM(spec BlockSpec, modelDim int) int {
	if spec.NSSM > 0 {
		return spec.NSSM
	}
	return modelDim
}

func effectiveS4DDiscretization(spec BlockSpec) string {
	switch strings.ToLower(strings.TrimSpace(spec.Discretization)) {
	case "", S4DDiscretizationZOH:
		return S4DDiscretizationZOH
	case S4DDiscretizationBilinear:
		return S4DDiscretizationBilinear
	default:
		return strings.ToLower(strings.TrimSpace(spec.Discretization))
	}
}

func s4dUsesAdvancedKernel(spec BlockSpec) bool {
	return spec.Bidirectional ||
		spec.NSSM != 0 ||
		spec.TrainableB ||
		effectiveS4DDiscretization(spec) != S4DDiscretizationZOH
}

func (cfg *ArchConfig) EffectiveFinalNorm() bool {
	return cfg == nil || cfg.FinalNorm == nil || *cfg.FinalNorm
}

func (t TrainingSpec) EffectiveWeightDecayPolicy() string {
	switch strings.ToLower(strings.TrimSpace(t.WeightDecayPolicy)) {
	case "", WeightDecayPolicyMatrixOnly:
		return WeightDecayPolicyMatrixOnly
	case WeightDecayPolicyAll:
		return WeightDecayPolicyAll
	default:
		return strings.ToLower(strings.TrimSpace(t.WeightDecayPolicy))
	}
}
