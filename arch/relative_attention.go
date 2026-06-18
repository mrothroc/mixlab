package arch

import "strings"

const (
	RelativeAttentionNone          = "none"
	RelativeAttentionDebertaP2CC2P = "deberta_p2c_c2p"

	RelativeAttentionParamPerBlockProjections = "per_block_projections"
	RelativeAttentionParamSharedQKReuse       = "shared_qk_reuse"

	SharedRelativeEmbeddingsWeightName = "shared_relative_embeddings"
	SharedRelativeNormScaleWeightName  = "shared_relative_norm_scale"
	SharedRelativeNormBiasWeightName   = "shared_relative_norm_bias"

	RelativeAttentionEmbeddingNormNone      = "none"
	RelativeAttentionEmbeddingNormLayerNorm = "layernorm"

	defaultRelativeAttentionWindow = 128
)

func normalizeRelativeAttention(raw string) string {
	return strings.ToLower(strings.TrimSpace(raw))
}

func relativeAttentionEnabled(spec BlockSpec) bool {
	return normalizeRelativeAttention(spec.RelativeAttention) == RelativeAttentionDebertaP2CC2P
}

func normalizeRelativeAttentionParameterization(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "":
		return RelativeAttentionParamPerBlockProjections
	default:
		return value
	}
}

func normalizeRelativeAttentionEmbeddingNorm(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "", "none", "off", "disabled", "false":
		return RelativeAttentionEmbeddingNormNone
	case "layernorm", "layer_norm", "ln":
		return RelativeAttentionEmbeddingNormLayerNorm
	default:
		return value
	}
}

func relativeAttentionUsesSharedQKReuse(spec BlockSpec) bool {
	return relativeAttentionEnabled(spec) &&
		normalizeRelativeAttentionParameterization(spec.RelativeAttentionParameterization) == RelativeAttentionParamSharedQKReuse
}

func relativeAttentionUsesPerBlockProjections(spec BlockSpec) bool {
	return relativeAttentionEnabled(spec) && !relativeAttentionUsesSharedQKReuse(spec)
}

func effectiveRelativeAttentionWindow(spec BlockSpec) int {
	if !relativeAttentionEnabled(spec) {
		return 0
	}
	if spec.RelativeAttentionWindow > 0 {
		return spec.RelativeAttentionWindow
	}
	return defaultRelativeAttentionWindow
}

type sharedRelativeAttentionPlan struct {
	Enabled     bool
	Window      int
	WeightIndex int
	Norm        string
	NormIndex   int
	NormEps     float32
}

func (p sharedRelativeAttentionPlan) Rows() int {
	if !p.Enabled {
		return 0
	}
	return 2*p.Window - 1
}

func newSharedRelativeAttentionPlan(blocks []BlockSpec) (sharedRelativeAttentionPlan, error) {
	plan := sharedRelativeAttentionPlan{WeightIndex: -1, NormIndex: -1, Norm: RelativeAttentionEmbeddingNormNone}
	for _, block := range blocks {
		if !relativeAttentionUsesSharedQKReuse(block) {
			continue
		}
		window := effectiveRelativeAttentionWindow(block)
		embeddingNorm := normalizeRelativeAttentionEmbeddingNorm(block.RelativeAttentionEmbeddingNorm)
		if !plan.Enabled {
			plan.Enabled = true
			plan.Window = window
			plan.Norm = embeddingNorm
			continue
		}
		if plan.Window != window {
			return plan, errSharedRelativeWindowMismatch(plan.Window, window)
		}
		if plan.Norm != embeddingNorm {
			return plan, &sharedRelativeEmbeddingNormMismatchError{first: plan.Norm, got: embeddingNorm}
		}
	}
	return plan, nil
}

func errSharedRelativeWindowMismatch(first, got int) error {
	return &sharedRelativeWindowMismatchError{first: first, got: got}
}

type sharedRelativeWindowMismatchError struct {
	first int
	got   int
}

func (e *sharedRelativeWindowMismatchError) Error() string {
	return "shared_qk_reuse relative_attention blocks must use the same effective relative_attention_window"
}

type sharedRelativeEmbeddingNormMismatchError struct {
	first string
	got   string
}

func (e *sharedRelativeEmbeddingNormMismatchError) Error() string {
	return "shared_qk_reuse relative_attention blocks must use the same relative_attention_embedding_norm"
}

func sharedRelativeAttentionWeightShapes(modelDim int, blocks []BlockSpec) ([]WeightMeta, error) {
	plan, err := newSharedRelativeAttentionPlan(blocks)
	if err != nil {
		return nil, err
	}
	if !plan.Enabled {
		return nil, nil
	}
	shapes := []WeightMeta{{Name: SharedRelativeEmbeddingsWeightName, Shape: []int{plan.Rows(), modelDim}}}
	if plan.Norm == RelativeAttentionEmbeddingNormLayerNorm {
		shapes = append(shapes,
			WeightMeta{Name: SharedRelativeNormScaleWeightName, Shape: []int{modelDim}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: SharedRelativeNormBiasWeightName, Shape: []int{modelDim}, InitZero: true},
		)
	}
	return shapes, nil
}

func sharedRelativeAttentionWeightCount(blocks []BlockSpec) (int, error) {
	plan, err := newSharedRelativeAttentionPlan(blocks)
	if err != nil {
		return 0, err
	}
	if !plan.Enabled {
		return 0, nil
	}
	if plan.Norm == RelativeAttentionEmbeddingNormLayerNorm {
		return 3, nil
	}
	return 1, nil
}
