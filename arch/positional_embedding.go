package arch

import (
	"fmt"
	"strings"
)

// Positional-embedding and Hugging Face export-format configuration. RoPE is the
// legacy default; learned_absolute adds a GPT-2-style WPE table and disables RoPE.

const (
	PositionalEmbeddingRope            = "rope"
	PositionalEmbeddingLearnedAbsolute = "learned_absolute"
	PositionalEmbeddingNone            = "none"

	HFExportFormatMixlab = "mixlab"
	HFExportFormatGPT2   = "gpt2"
)

// EffectiveEmbeddingDropout returns training-time dropout on the summed
// embedding stream before block 0.
func (c *ArchConfig) EffectiveEmbeddingDropout() float32 {
	if c == nil {
		return 0
	}
	return c.EmbeddingDropout
}

// EffectiveMaxPositions returns the learned-position table length, defaulting
// to seq_len for configs that do not need a longer generation context.
func (c *ArchConfig) EffectiveMaxPositions() int {
	if c == nil {
		return 0
	}
	if c.MaxPositions > 0 {
		return c.MaxPositions
	}
	return c.SeqLen
}

// EffectivePositionalEmbedding returns the model-level positional embedding
// mode. Empty configs preserve the legacy RoPE path.
func (c *ArchConfig) EffectivePositionalEmbedding() string {
	if c == nil {
		return PositionalEmbeddingRope
	}
	return normalizePositionalEmbedding(c.PositionalEmbedding)
}

func normalizePositionalEmbedding(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", PositionalEmbeddingRope:
		return PositionalEmbeddingRope
	case "learned", "learned_absolute", "learned-absolute":
		return PositionalEmbeddingLearnedAbsolute
	case PositionalEmbeddingNone:
		return PositionalEmbeddingNone
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

// EffectiveHFExportFormat returns the requested Hugging Face export format.
// Empty configs keep the existing Mixlab custom-code export.
func (c *ArchConfig) EffectiveHFExportFormat() string {
	if c == nil {
		return HFExportFormatMixlab
	}
	return normalizeHFExportFormat(c.HFExportFormat)
}

func normalizeHFExportFormat(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "", HFExportFormatMixlab:
		return HFExportFormatMixlab
	case HFExportFormatGPT2:
		return HFExportFormatGPT2
	default:
		return strings.ToLower(strings.TrimSpace(v))
	}
}

func validatePositionalAndExportConfig(cfg *ArchConfig, source string) error {
	posMode := normalizePositionalEmbedding(cfg.PositionalEmbedding)
	switch posMode {
	case PositionalEmbeddingRope, PositionalEmbeddingLearnedAbsolute, PositionalEmbeddingNone:
		if strings.TrimSpace(cfg.PositionalEmbedding) != "" {
			cfg.PositionalEmbedding = posMode
		}
	default:
		return fmt.Errorf("config %q has invalid positional_embedding=%q (must be \"rope\", \"learned_absolute\", or \"none\")", source, cfg.PositionalEmbedding)
	}
	if cfg.MaxPositions < 0 {
		return fmt.Errorf("config %q has invalid max_positions=%d (must be >= 0)", source, cfg.MaxPositions)
	}
	if cfg.EffectiveMaxPositions() < cfg.SeqLen {
		return fmt.Errorf("config %q has invalid max_positions=%d (must be >= seq_len=%d)", source, cfg.MaxPositions, cfg.SeqLen)
	}
	if cfg.EmbeddingDropout < 0 || cfg.EmbeddingDropout > 1 {
		return fmt.Errorf("config %q has invalid embedding_dropout=%g (must be in [0,1])", source, cfg.EmbeddingDropout)
	}
	exportFormat := normalizeHFExportFormat(cfg.HFExportFormat)
	switch exportFormat {
	case HFExportFormatMixlab, HFExportFormatGPT2:
		if strings.TrimSpace(cfg.HFExportFormat) != "" {
			cfg.HFExportFormat = exportFormat
		}
	default:
		return fmt.Errorf("config %q has invalid hf_export_format=%q (must be \"mixlab\" or \"gpt2\")", source, cfg.HFExportFormat)
	}
	return nil
}

func validateBlockPositionalEmbedding(cfg *ArchConfig, b BlockSpec, source, groupName string, idx int) error {
	if cfg == nil || blockTypeKey(b) != "plain" {
		return nil
	}
	if cfg.EffectivePositionalEmbedding() == PositionalEmbeddingLearnedAbsolute {
		if b.RopeDims != 0 {
			return fmt.Errorf("config %q %s[%d] type=plain cannot combine positional_embedding=\"learned_absolute\" with rope_dims", source, groupName, idx)
		}
		if strings.TrimSpace(b.RopeConvention) != "" {
			return fmt.Errorf("config %q %s[%d] type=plain cannot combine positional_embedding=\"learned_absolute\" with rope_convention", source, groupName, idx)
		}
		if relativeAttentionEnabled(b) {
			return fmt.Errorf("config %q %s[%d] type=plain cannot combine positional_embedding=\"learned_absolute\" with relative_attention", source, groupName, idx)
		}
	}
	return nil
}

func positionalEmbeddingWeightShapes(modelDim, maxPositions int, positionalEmbedding string) []WeightMeta {
	if normalizePositionalEmbedding(positionalEmbedding) != PositionalEmbeddingLearnedAbsolute {
		return nil
	}
	if maxPositions <= 0 {
		return nil
	}
	return []WeightMeta{{Name: "position_embeddings", Shape: []int{maxPositions, modelDim}}}
}

func emitLearnedPositionEmbeddingIR(prog *Program, embedState string, B, T, D, wi, maxPositions int) (string, int, error) {
	if maxPositions < T {
		return embedState, wi, fmt.Errorf("max_positions=%d must be >= seq_len=%d", maxPositions, T)
	}
	prog.Arange(0, T, "position_ids")
	prog.Embed(weightName(wi), "position_ids", "position_embed")
	prog.Reshape("position_embed", []int{1, T, D}, "position_embed_1td")
	prog.Full([]int{B, 1, 1}, 1, "position_embed_broadcast")
	prog.Mul("position_embed_1td", "position_embed_broadcast", "position_embed_btd")
	prog.Add(embedState, "position_embed_btd", "x_embed_pos")
	return "x_embed_pos", wi + 1, nil
}
