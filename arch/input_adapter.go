package arch

import (
	"fmt"
	"strings"
)

const (
	InputAdapterTokenEmbedding = "token_embedding"
	InputAdapterLinearFrames   = "linear_frames"

	InputAdapterNormNone      = "none"
	InputAdapterNormLayerNorm = "layernorm"

	inputAdapterLowDimLayerNormWarningMaxFeatureDim = 4
)

// InputAdapterSpec selects how model inputs become [B,T,model_dim] hidden
// states. Omission preserves the historical token embedding path.
type InputAdapterSpec struct {
	Kind       string `json:"kind,omitempty"`
	FeatureDim int    `json:"feature_dim,omitempty"`
	Bias       *bool  `json:"bias,omitempty"`
	Norm       string `json:"norm,omitempty"`
}

func normalizeInputAdapterKind(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", InputAdapterTokenEmbedding:
		return InputAdapterTokenEmbedding
	case InputAdapterLinearFrames:
		return InputAdapterLinearFrames
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func normalizeInputAdapterNorm(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", InputAdapterNormNone:
		return InputAdapterNormNone
	case "layer_norm", InputAdapterNormLayerNorm:
		return InputAdapterNormLayerNorm
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func (cfg *ArchConfig) EffectiveInputAdapterKind() string {
	if cfg == nil || cfg.InputAdapter == nil {
		return InputAdapterTokenEmbedding
	}
	return normalizeInputAdapterKind(cfg.InputAdapter.Kind)
}

func (cfg *ArchConfig) LinearFramesEnabled() bool {
	return cfg != nil && cfg.EffectiveInputAdapterKind() == InputAdapterLinearFrames
}

// InputFeatureDim returns the per-timestep feature width for continuous
// adapters. Token inputs do not have a public continuous feature dimension.
func (cfg *ArchConfig) InputFeatureDim() int {
	if cfg == nil || !cfg.LinearFramesEnabled() || cfg.InputAdapter == nil {
		return 0
	}
	return cfg.InputAdapter.FeatureDim
}

func (cfg *ArchConfig) EffectiveInputAdapterBias() bool {
	if cfg == nil || cfg.InputAdapter == nil || cfg.InputAdapter.Bias == nil {
		return true
	}
	return *cfg.InputAdapter.Bias
}

func (cfg *ArchConfig) EffectiveInputAdapterNorm() string {
	if cfg == nil || cfg.InputAdapter == nil {
		return InputAdapterNormNone
	}
	return normalizeInputAdapterNorm(cfg.InputAdapter.Norm)
}

func inputAdapterWarnings(cfg *ArchConfig, source string) []string {
	if cfg == nil || !cfg.LinearFramesEnabled() || cfg.InputAdapter == nil {
		return nil
	}
	if cfg.EffectiveInputAdapterNorm() != InputAdapterNormLayerNorm ||
		cfg.InputAdapter.FeatureDim > inputAdapterLowDimLayerNormWarningMaxFeatureDim {
		return nil
	}
	return []string{fmt.Sprintf(
		"WARN: config %q uses input_adapter.kind=%q with feature_dim=%d and norm=%q; post-projection LayerNorm can discard input magnitude at initialization (feature_dim=1 is exactly sign-only while the projection bias is zero). Prefer norm=%q for magnitude-bearing raw signals.",
		source, InputAdapterLinearFrames, cfg.InputAdapter.FeatureDim, InputAdapterNormLayerNorm, InputAdapterNormNone,
	)}
}

func linearFramesExtraWeightShapes(cfg *ArchConfig) []WeightMeta {
	if cfg == nil || !cfg.LinearFramesEnabled() {
		return nil
	}
	var out []WeightMeta
	if cfg.EffectiveInputAdapterBias() {
		out = append(out, WeightMeta{
			Name: "input_adapter_bias", Shape: []int{cfg.ModelDim}, InitZero: true,
		})
	}
	if cfg.EffectiveInputAdapterNorm() == InputAdapterNormLayerNorm {
		out = append(out,
			WeightMeta{Name: "input_adapter_norm_scale", Shape: []int{cfg.ModelDim}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "input_adapter_norm_bias", Shape: []int{cfg.ModelDim}, InitZero: true},
		)
	}
	return out
}

func validateInputAdapter(cfg *ArchConfig, source string) error {
	if cfg == nil {
		return fmt.Errorf("config %q is nil", source)
	}
	kind := cfg.EffectiveInputAdapterKind()
	switch kind {
	case InputAdapterTokenEmbedding:
		if cfg.InputAdapter != nil {
			cfg.InputAdapter.Kind = InputAdapterTokenEmbedding
			if cfg.InputAdapter.FeatureDim != 0 {
				return fmt.Errorf("config %q input_adapter.feature_dim is valid only with kind=%q", source, InputAdapterLinearFrames)
			}
			if cfg.InputAdapter.Bias != nil {
				return fmt.Errorf("config %q input_adapter.bias is valid only with kind=%q", source, InputAdapterLinearFrames)
			}
			if norm := normalizeInputAdapterNorm(cfg.InputAdapter.Norm); norm != InputAdapterNormNone {
				return fmt.Errorf("config %q input_adapter.norm is valid only with kind=%q", source, InputAdapterLinearFrames)
			}
		}
		return nil
	case InputAdapterLinearFrames:
	default:
		return fmt.Errorf(
			"config %q has invalid input_adapter.kind=%q (must be %q or %q)",
			source, kind, InputAdapterTokenEmbedding, InputAdapterLinearFrames,
		)
	}

	spec := cfg.InputAdapter
	spec.Kind = InputAdapterLinearFrames
	spec.Norm = normalizeInputAdapterNorm(spec.Norm)
	if spec.FeatureDim <= 0 {
		return fmt.Errorf("config %q input_adapter.feature_dim=%d must be > 0 for kind=%q", source, spec.FeatureDim, InputAdapterLinearFrames)
	}
	switch spec.Norm {
	case InputAdapterNormNone, InputAdapterNormLayerNorm:
	default:
		return fmt.Errorf("config %q input_adapter.norm=%q must be %q or %q", source, spec.Norm, InputAdapterNormNone, InputAdapterNormLayerNorm)
	}
	if cfg.VocabSize != 0 {
		return fmt.Errorf("config %q input_adapter.kind=%q requires vocab_size to be omitted or 0; got %d", source, InputAdapterLinearFrames, cfg.VocabSize)
	}
	if cfg.Training.EffectiveObjective() != ObjectiveClassification || cfg.Training.Classification == nil {
		return fmt.Errorf("config %q input_adapter.kind=%q supports training.objective=%q only in v1", source, InputAdapterLinearFrames, ObjectiveClassification)
	}
	if cfg.TieEmbeddings {
		return fmt.Errorf("config %q input_adapter.kind=%q does not use tie_embeddings", source, InputAdapterLinearFrames)
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use token embedding feature channels", source, InputAdapterLinearFrames)
	}
	if cfg.RCEquivarianceEnabled() || cfg.Training.ReverseComplementProb > 0 {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use reverse-complement token features", source, InputAdapterLinearFrames)
	}
	if cfg.MTP != nil || cfg.MLMHead != "" {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use language-model heads or MTP", source, InputAdapterLinearFrames)
	}
	if cfg.Training.ExampleFramingEnabled() || cfg.Training.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("config %q input_adapter.kind=%q does not use token framing or segment masks", source, InputAdapterLinearFrames)
	}
	return nil
}
