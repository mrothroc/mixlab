package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	defaultTTTMLPChunkSize       = 16
	defaultTTTMLPInnerHiddenMult = 4.0
	defaultTTTMLPInnerLRBase     = 0.1
	defaultTTTMLPInnerLRInit     = 0.01
	defaultTTTMLPInnerLRWarmup   = 5000
	defaultTTTMLPConvKernel      = 4
)

func effectiveTTTMLPChunkSize(spec BlockSpec) int {
	if spec.chunkSizeSet || spec.ChunkSize > 0 {
		return spec.ChunkSize
	}
	return defaultTTTMLPChunkSize
}

func effectiveTTTMLPInnerHiddenMult(spec BlockSpec) float64 {
	if spec.innerHiddenMultSet || spec.InnerHiddenMult > 0 {
		return spec.InnerHiddenMult
	}
	return defaultTTTMLPInnerHiddenMult
}

func effectiveTTTMLPInnerLRBase(spec BlockSpec) float64 {
	if spec.innerLRBaseSet || spec.InnerLRBase > 0 {
		return spec.InnerLRBase
	}
	return defaultTTTMLPInnerLRBase
}

func effectiveTTTMLPInnerLRInit(spec BlockSpec) float64 {
	if spec.innerLRInitSet || spec.InnerLRInit > 0 {
		return spec.InnerLRInit
	}
	return defaultTTTMLPInnerLRInit
}

func effectiveTTTMLPInnerLRWarmupSteps(spec BlockSpec) int {
	if spec.InnerLRWarmupSteps != nil {
		return *spec.InnerLRWarmupSteps
	}
	return defaultTTTMLPInnerLRWarmup
}

func effectiveTTTMLPInnerHiddenDim(spec BlockSpec, modelDim int) (int, error) {
	if spec.Heads <= 0 || modelDim <= 0 || modelDim%spec.Heads != 0 {
		return 0, fmt.Errorf("ttt_mlp requires model_dim divisible by heads")
	}
	headDim := modelDim / spec.Heads
	hidden := int(math.Round(float64(headDim) * effectiveTTTMLPInnerHiddenMult(spec)))
	if hidden <= 0 {
		return 0, fmt.Errorf("ttt_mlp inner_hidden_mult produces an empty hidden layer")
	}
	return hidden, nil
}

func validateTTTMLPPolicy(cfg *ArchConfig, source string) error {
	first := -1
	for i, block := range cfg.Blocks {
		if blockTypeKey(block) == "ttt_mlp" {
			first = i
			break
		}
	}
	if first < 0 {
		return nil
	}

	field := fmt.Sprintf("config %q blocks[%d] type=ttt_mlp", source, first)
	if objective := cfg.Training.EffectiveObjective(); objective != ObjectiveCausal && objective != ObjectiveClassification {
		return fmt.Errorf("%s supports causal only for language modeling, plus sequence classification, in v1", field)
	}
	if cfg.UNet {
		return fmt.Errorf("%s does not support unet in v1", field)
	}
	if cfg.ParallelResidual {
		return fmt.Errorf("%s does not support parallel_residual in v1", field)
	}
	for i, block := range cfg.Blocks {
		if block.ParallelGroup > 0 || block.ParallelResidual != nil {
			return fmt.Errorf("%s does not support blocks[%d] parallel grouping in v1", field, i)
		}
		if blockTypeKey(block) == "custom" {
			return fmt.Errorf("%s does not support custom execution blocks in v1", field)
		}
	}
	if len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 {
		return fmt.Errorf("%s does not support recurrence or recurrence_phases in v1", field)
	}
	if cfg.Training.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("%s does not support attention segment packing in v1", field)
	}
	if cfg.MTP != nil || cfg.Training.FirstByteMask || cfg.Training.Distillation != nil ||
		cfg.Training.Data2VecActive() || cfg.Training.ExampleFramingEnabled() ||
		cfg.Training.MinimalPair != nil || cfg.Training.RTD != nil ||
		cfg.Training.WordStructuralObjective != nil || cfg.Training.Invariance != nil ||
		cfg.Training.PLLMargin != nil {
		return fmt.Errorf("%s supports the primary causal loss only in v1", field)
	}
	if strings.TrimSpace(cfg.LayerAggregation) != "" && normalizeLayerAggregation(cfg.LayerAggregation) != LayerAggregationNone {
		return fmt.Errorf("%s does not support layer aggregation in v1", field)
	}
	return nil
}
