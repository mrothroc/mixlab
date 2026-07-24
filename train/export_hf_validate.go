package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

func validateHFExportConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return fmt.Errorf("unsupported HF export: nil config")
	}
	if cfg.RCEquivarianceEnabled() {
		return unsupportedHFExport("rc_equivariant", "shared reverse-complement branches are native-only in v1")
	}
	if cfg.BlockScales {
		return unsupportedHFExport("block_scales", "core HF export does not yet support block scale tensors")
	}
	if cfg.ResidMix {
		return unsupportedHFExport("resid_mix", "residual mixing is not part of the walking skeleton")
	}
	if cfg.ParallelResidual {
		return unsupportedHFExport("parallel_residual", "parallel residual export is planned for a later release")
	}
	if cfg.UNet {
		return unsupportedHFExport("unet", "U-Net export is not part of the walking skeleton")
	}
	if cfg.SmearEmbeddings {
		return unsupportedHFExport("smear_embeddings", "embedding smear export is not part of the walking skeleton")
	}
	if cfg.MTP != nil {
		return unsupportedHFExport("mtp", "MTP is training-only and has no core HF export semantics")
	}
	if cfg.Backout != nil {
		return unsupportedHFExport("backout", "backout export is not part of the walking skeleton")
	}
	if len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 {
		return unsupportedHFExport("recurrence", "weight sharing and recurrence phases are planned for later HF coverage")
	}
	if cfg.Training.EffectiveObjective() == arch.ObjectiveBlockDiffusion {
		return unsupportedHFExport("training.objective", "pure block_diffusion has native-only generation semantics; export the causal view of a hybrid checkpoint instead")
	}
	switch cfg.Training.EffectiveObjective() {
	case "causal", "hybrid", "mlm", "mntp":
	default:
		return unsupportedHFExport("training.objective", fmt.Sprintf("unknown objective %q", cfg.Training.EffectiveObjective()))
	}
	if cfg.Training.FirstByteMask {
		return unsupportedHFExport("training.first_byte_mask", "first-byte masked loss is training-only")
	}
	if cfg.Eval != nil && cfg.EffectiveEvalSpec().LegalChunkSGDEnabled() {
		return unsupportedHFExport("eval.ttt_mode", "eval-time TTT is not represented in the exported HF model")
	}
	switch cfg.EffectiveLayerAggregation() {
	case "", "none", "dwa":
	default:
		return unsupportedHFExport("layer_aggregation", fmt.Sprintf("unsupported layer_aggregation %q", cfg.LayerAggregation))
	}

	for i, block := range cfg.Blocks {
		field := fmt.Sprintf("blocks[%d]", i)
		if strings.TrimSpace(block.WeightGroup) != "" {
			return unsupportedHFExport(field+".weight_group", "weight groups are not part of the walking skeleton")
		}
		if block.ParallelResidual != nil {
			return unsupportedHFExport(field+".parallel_residual", "parallel residual export is planned for a later release")
		}
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			if _, err := normalizeHFExportKVHeads(block.Heads, block.KVHeads); err != nil {
				return unsupportedHFExport(field+".kv_heads", err.Error())
			}
			if block.KVSource > 0 {
				return unsupportedHFExport(field+".kv_source", "KV sharing export is planned for a later release")
			}
			if block.SkipAttention {
				return unsupportedHFExport(field+".skip_attention", "skip_attention is not a deployable HF forward path")
			}
			relAttention := strings.ToLower(strings.TrimSpace(block.RelativeAttention))
			switch relAttention {
			case "", "none", "deberta_p2c_c2p":
			default:
				return unsupportedHFExport(field+".relative_attention", fmt.Sprintf("unsupported relative_attention %q", block.RelativeAttention))
			}
			switch hfRelativeAttentionParameterization(block) {
			case "per_block_projections":
			case "shared_qk_reuse":
				if !relativeAttentionEnabledForHF(block) {
					return unsupportedHFExport(field+".relative_attention_parameterization", "shared_qk_reuse requires relative_attention=deberta_p2c_c2p")
				}
			default:
				return unsupportedHFExport(field+".relative_attention_parameterization", fmt.Sprintf("unsupported relative_attention_parameterization %q", block.RelativeAttentionParameterization))
			}
			mask := strings.ToLower(strings.TrimSpace(block.AttentionMask))
			switch mask {
			case "", "causal", "bidirectional", "none":
			default:
				return unsupportedHFExport(field+".attention_mask", fmt.Sprintf("invalid attention mask %q", block.AttentionMask))
			}
			if block.WindowSize > 0 && mask != "" && mask != "causal" {
				return unsupportedHFExport(field+".window_size", "windowed attention export requires causal attention")
			}
			switch hfPlainFFNActivation(block) {
			case "silu", "gelu", "gelu_new", "geglu", "swiglu":
			default:
				return unsupportedHFExport(field+".ffn_activation", fmt.Sprintf("unsupported plain ffn_activation %q", block.FFNActivation))
			}
		case "swiglu", "geglu":
			// supported
		case "mlp":
			switch strings.ToLower(strings.TrimSpace(block.Activation)) {
			case "", "silu", "gelu", "relu", "leaky_relu_sq":
			default:
				return unsupportedHFExport(field+".activation", fmt.Sprintf("unsupported MLP activation %q", block.Activation))
			}
		case "moe":
			if err := validateHFExportMoEBlock(field, block); err != nil {
				return err
			}
		case "ttt_mlp":
			if err := validateHFTTTMLPComposition(cfg, field, block); err != nil {
				return err
			}
		default:
			capability := hfExportBlockCapability(block)
			return unsupportedHFExport(field+".type", fmt.Sprintf("%s: %s", capability.Feature, capability.Reason))
		}
	}
	return nil
}

func validateHFTTTMLPComposition(cfg *ArchConfig, field string, block BlockSpec) error {
	if cfg.Training.EffectiveObjective() != arch.ObjectiveCausal {
		return unsupportedHFExport(field+".type", "ttt_mlp export requires a causal objective")
	}
	if cfg.EffectivePositionalEmbedding() != arch.PositionalEmbeddingRope {
		return unsupportedHFExport(field+".type", "ttt_mlp export requires positional_embedding=rope")
	}
	if block.Heads <= 0 || cfg.ModelDim%block.Heads != 0 || (cfg.ModelDim/block.Heads)%2 != 0 {
		return unsupportedHFExport(field+".heads", "ttt_mlp requires model_dim divisible by heads with an even head_dim")
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return unsupportedHFExport("feature_embeddings", "ttt_mlp cached export does not support embedding feature channels")
	}
	for i, candidate := range cfg.Blocks {
		switch strings.ToLower(strings.TrimSpace(candidate.Type)) {
		case "ttt_mlp", "swiglu", "geglu", "mlp":
		default:
			return unsupportedHFExport(
				fmt.Sprintf("blocks[%d].type", i),
				"ttt_mlp cached export supports only ttt_mlp plus pointwise swiglu/geglu/mlp blocks",
			)
		}
	}
	return nil
}

func validateHFExportMoEBlock(field string, block BlockSpec) error {
	if strings.ToLower(strings.TrimSpace(block.Router)) != "" && strings.ToLower(strings.TrimSpace(block.Router)) != "linear" {
		return unsupportedHFExport(field+".router", fmt.Sprintf("unsupported MoE router %q", block.Router))
	}
	if block.NumExperts <= 0 {
		return unsupportedHFExport(field+".num_experts", "moe requires num_experts > 0")
	}
	topK := block.TopK
	if topK <= 0 {
		if block.NumExperts <= 1 {
			topK = 1
		} else {
			topK = 2
		}
	}
	if topK < 1 || topK > block.NumExperts {
		return unsupportedHFExport(field+".top_k", fmt.Sprintf("top_k must be in [1,num_experts], got %d for %d experts", topK, block.NumExperts))
	}
	expert := BlockSpec{Type: "swiglu"}
	if block.ExpertBlock != nil {
		expert = *block.ExpertBlock
	}
	switch strings.ToLower(strings.TrimSpace(expert.Type)) {
	case "swiglu", "geglu":
		return nil
	case "mlp":
		switch strings.ToLower(strings.TrimSpace(expert.Activation)) {
		case "", "silu", "gelu", "relu", "leaky_relu_sq":
			return nil
		default:
			return unsupportedHFExport(field+".expert_block.activation", fmt.Sprintf("unsupported MoE MLP activation %q", expert.Activation))
		}
	default:
		return unsupportedHFExport(field+".expert_block.type", fmt.Sprintf("unsupported MoE expert block type %q", expert.Type))
	}
}

func unsupportedHFExport(field, reason string) error {
	return fmt.Errorf("unsupported HF export feature %s: %s", field, reason)
}

func normalizeHFExportKVHeads(heads, kvHeads int) (int, error) {
	if heads <= 0 {
		return 0, fmt.Errorf("heads must be > 0")
	}
	if kvHeads == 0 {
		return heads, nil
	}
	if kvHeads < 0 {
		return 0, fmt.Errorf("kv_heads must be > 0 when set")
	}
	if heads%kvHeads != 0 {
		return 0, fmt.Errorf("heads %% kv_heads must be 0 (heads=%d kv_heads=%d)", heads, kvHeads)
	}
	return kvHeads, nil
}
