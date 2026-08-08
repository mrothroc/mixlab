package arch

import "strings"

const (
	S4DDiscretizationZOH      = "zoh"
	S4DDiscretizationBilinear = "bilinear"
	DefaultS4DSobolevLR       = 0.01

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
