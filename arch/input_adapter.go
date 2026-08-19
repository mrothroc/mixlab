package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	InputAdapterTokenEmbedding    = "token_embedding"
	InputAdapterLinearFrames      = "linear_frames"
	InputAdapterDiscreteCodebooks = "discrete_codebooks"

	InputAdapterFusionAttentionMLP = "attention_mlp"
	InputAdapterFusionMean         = "mean"

	InputAdapterNormNone      = "none"
	InputAdapterNormLayerNorm = "layernorm"

	inputAdapterLowDimLayerNormWarningMaxFeatureDim = 4
)

// InputAdapterSpec selects how model inputs become [B,T,model_dim] hidden
// states. Omission preserves the historical token embedding path.
type InputAdapterSpec struct {
	Kind              string `json:"kind,omitempty"`
	FeatureDim        int    `json:"feature_dim,omitempty"`
	Bias              *bool  `json:"bias,omitempty"`
	Norm              string `json:"norm,omitempty"`
	NumCodebooks      int    `json:"num_codebooks,omitempty"`
	CodebookVocabSize int    `json:"codebook_vocab_size,omitempty"`
	Fusion            string `json:"fusion,omitempty"`
	FusionHiddenDim   int    `json:"fusion_hidden_dim,omitempty"`
}

func normalizeInputAdapterKind(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", InputAdapterTokenEmbedding:
		return InputAdapterTokenEmbedding
	case InputAdapterLinearFrames:
		return InputAdapterLinearFrames
	case InputAdapterDiscreteCodebooks:
		return InputAdapterDiscreteCodebooks
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func normalizeInputAdapterFusion(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", InputAdapterFusionAttentionMLP:
		return InputAdapterFusionAttentionMLP
	case InputAdapterFusionMean:
		return InputAdapterFusionMean
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

func (cfg *ArchConfig) DiscreteCodebooksEnabled() bool {
	return cfg != nil && cfg.EffectiveInputAdapterKind() == InputAdapterDiscreteCodebooks
}

func (cfg *ArchConfig) EffectiveCodebookFusion() string {
	if cfg == nil || cfg.InputAdapter == nil {
		return ""
	}
	return normalizeInputAdapterFusion(cfg.InputAdapter.Fusion)
}

func (cfg *ArchConfig) EffectiveCodebookFusionHiddenDim() int {
	if cfg == nil || !cfg.DiscreteCodebooksEnabled() || cfg.InputAdapter == nil {
		return 0
	}
	if cfg.InputAdapter.FusionHiddenDim > 0 {
		return cfg.InputAdapter.FusionHiddenDim
	}
	return cfg.ModelDim
}

func (cfg *ArchConfig) CodebookEmbeddingRows() int {
	if cfg == nil || !cfg.DiscreteCodebooksEnabled() || cfg.InputAdapter == nil {
		return 0
	}
	return cfg.InputAdapter.NumCodebooks * cfg.InputAdapter.CodebookVocabSize
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
		out = append(out, linearBiasWeightMeta(
			"input_adapter_bias", cfg.InputAdapter.FeatureDim, cfg.ModelDim,
		))
	}
	if cfg.EffectiveInputAdapterNorm() == InputAdapterNormLayerNorm {
		out = append(out,
			WeightMeta{Name: "input_adapter_norm_scale", Shape: []int{cfg.ModelDim}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "input_adapter_norm_bias", Shape: []int{cfg.ModelDim}, InitZero: true},
		)
	}
	return out
}

func discreteCodebookExtraWeightShapes(cfg *ArchConfig) []WeightMeta {
	if cfg == nil || !cfg.DiscreteCodebooksEnabled() || cfg.InputAdapter == nil {
		return nil
	}
	var out []WeightMeta
	if cfg.EffectiveCodebookFusion() == InputAdapterFusionAttentionMLP {
		hidden := cfg.EffectiveCodebookFusionHiddenDim()
		out = append(out,
			WeightMeta{Name: "input_adapter_codebook_attn_w1", Shape: []int{cfg.ModelDim, hidden}, InitMode: "torch_linear_uniform"},
			WeightMeta{Name: "input_adapter_codebook_attn_b1", Shape: []int{hidden}, InitMode: "torch_linear_bias_uniform", PyTorchLinearFanIn: cfg.ModelDim},
			WeightMeta{Name: "input_adapter_codebook_attn_w2", Shape: []int{hidden, 1}, InitMode: "torch_linear_uniform"},
		)
	}
	if cfg.EffectiveInputAdapterNorm() == InputAdapterNormLayerNorm {
		out = append(out,
			WeightMeta{Name: "input_adapter_norm_scale", Shape: []int{cfg.ModelDim}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "input_adapter_norm_bias", Shape: []int{cfg.ModelDim}, InitZero: true},
		)
	}
	return out
}

func validateInputAdapterFeatureExclusions(cfg *ArchConfig, source, kind string) error {
	if cfg.VocabSize != 0 {
		return fmt.Errorf("config %q input_adapter.kind=%q requires vocab_size to be omitted or 0; got %d", source, kind, cfg.VocabSize)
	}
	if cfg.Training.EffectiveObjective() != ObjectiveClassification || cfg.Training.Classification == nil {
		return fmt.Errorf("config %q input_adapter.kind=%q supports training.objective=%q only in v1", source, kind, ObjectiveClassification)
	}
	if cfg.TieEmbeddings {
		return fmt.Errorf("config %q input_adapter.kind=%q does not use tie_embeddings", source, kind)
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use token embedding feature channels", source, kind)
	}
	if cfg.RCEquivarianceEnabled() || cfg.Training.ReverseComplementProb > 0 {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use reverse-complement token features", source, kind)
	}
	if cfg.MTP != nil || cfg.MLMHead != "" {
		return fmt.Errorf("config %q input_adapter.kind=%q cannot use language-model heads or MTP", source, kind)
	}
	if cfg.Training.ExampleFramingEnabled() || cfg.Training.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("config %q input_adapter.kind=%q does not use token framing or segment masks", source, kind)
	}
	return nil
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
			if cfg.InputAdapter.NumCodebooks != 0 || cfg.InputAdapter.CodebookVocabSize != 0 ||
				strings.TrimSpace(cfg.InputAdapter.Fusion) != "" || cfg.InputAdapter.FusionHiddenDim != 0 {
				return fmt.Errorf("config %q codebook input_adapter fields require kind=%q", source, InputAdapterDiscreteCodebooks)
			}
		}
		return nil
	case InputAdapterLinearFrames:
		if cfg.InputAdapter.NumCodebooks != 0 || cfg.InputAdapter.CodebookVocabSize != 0 ||
			strings.TrimSpace(cfg.InputAdapter.Fusion) != "" || cfg.InputAdapter.FusionHiddenDim != 0 {
			return fmt.Errorf("config %q codebook input_adapter fields require kind=%q", source, InputAdapterDiscreteCodebooks)
		}
	case InputAdapterDiscreteCodebooks:
	default:
		return fmt.Errorf(
			"config %q has invalid input_adapter.kind=%q (must be %q, %q, or %q)",
			source, kind, InputAdapterTokenEmbedding, InputAdapterLinearFrames, InputAdapterDiscreteCodebooks,
		)
	}

	spec := cfg.InputAdapter
	spec.Norm = normalizeInputAdapterNorm(spec.Norm)
	if kind == InputAdapterDiscreteCodebooks {
		spec.Kind = InputAdapterDiscreteCodebooks
		spec.Fusion = normalizeInputAdapterFusion(spec.Fusion)
		if spec.FeatureDim != 0 || spec.Bias != nil {
			return fmt.Errorf("config %q input_adapter.feature_dim/bias are valid only with kind=%q", source, InputAdapterLinearFrames)
		}
		if spec.NumCodebooks < 1 {
			return fmt.Errorf("config %q input_adapter.num_codebooks=%d must be >= 1", source, spec.NumCodebooks)
		}
		if spec.CodebookVocabSize < 2 {
			return fmt.Errorf("config %q input_adapter.codebook_vocab_size=%d must be >= 2", source, spec.CodebookVocabSize)
		}
		if int64(spec.NumCodebooks)*int64(spec.CodebookVocabSize) > math.MaxInt32 {
			return fmt.Errorf("config %q input_adapter num_codebooks*codebook_vocab_size exceeds int32 indexing", source)
		}
		switch spec.Fusion {
		case InputAdapterFusionAttentionMLP:
			if spec.FusionHiddenDim < 0 {
				return fmt.Errorf("config %q input_adapter.fusion_hidden_dim=%d must be >= 0", source, spec.FusionHiddenDim)
			}
		case InputAdapterFusionMean:
			if spec.FusionHiddenDim != 0 {
				return fmt.Errorf("config %q input_adapter.fusion_hidden_dim is valid only with fusion=%q", source, InputAdapterFusionAttentionMLP)
			}
		default:
			return fmt.Errorf("config %q input_adapter.fusion=%q must be %q or %q", source, spec.Fusion, InputAdapterFusionAttentionMLP, InputAdapterFusionMean)
		}
		switch spec.Norm {
		case InputAdapterNormNone, InputAdapterNormLayerNorm:
		default:
			return fmt.Errorf("config %q input_adapter.norm=%q must be %q or %q", source, spec.Norm, InputAdapterNormNone, InputAdapterNormLayerNorm)
		}
		return validateInputAdapterFeatureExclusions(cfg, source, kind)
	}

	spec.Kind = InputAdapterLinearFrames
	if spec.FeatureDim <= 0 {
		return fmt.Errorf("config %q input_adapter.feature_dim=%d must be > 0 for kind=%q", source, spec.FeatureDim, InputAdapterLinearFrames)
	}
	switch spec.Norm {
	case InputAdapterNormNone, InputAdapterNormLayerNorm:
	default:
		return fmt.Errorf("config %q input_adapter.norm=%q must be %q or %q", source, spec.Norm, InputAdapterNormNone, InputAdapterNormLayerNorm)
	}
	return validateInputAdapterFeatureExclusions(cfg, source, kind)
}
