package train

import "strings"

type hfExportSupportStatus string

const (
	hfExportSupported    hfExportSupportStatus = "supported"
	hfExportGated        hfExportSupportStatus = "gated"
	hfExportUnsupported  hfExportSupportStatus = "unsupported"
	hfExportTrainingOnly hfExportSupportStatus = "training_only"
)

type hfExportCapability struct {
	Feature string
	Status  hfExportSupportStatus
	Reason  string
}

func hfExportCapabilities() []hfExportCapability {
	return []hfExportCapability{
		{Feature: "plain", Status: hfExportSupported, Reason: "Core attention export with RoPE, GQA, qk_norm, qk_gain, XSA, sparse attention gates, masks, and causal windowing."},
		{Feature: "plain.attn_bias", Status: hfExportSupported, Reason: "Q/K/V/O projection biases are mirrored in the generated PyTorch template."},
		{Feature: "plain.attn_value_gate", Status: hfExportSupported, Reason: "Value-projection attention gates are mirrored before the output projection in the generated PyTorch template."},
		{Feature: "plain.attn_post_norm", Status: hfExportSupported, Reason: "Attention post-norm can inherit legacy after-output placement or explicitly run before the output projection."},
		{Feature: "plain.qk_norm", Status: hfExportSupported, Reason: "Learned Q/K RMSNorm scales are mirrored in the generated PyTorch template."},
		{Feature: "plain.xsa", Status: hfExportSupported, Reason: "XSA output projection is mirrored in the generated PyTorch template."},
		{Feature: "plain.sparse_attn_gate", Status: hfExportSupported, Reason: "Sparse per-head attention gates are mirrored in the generated PyTorch template."},
		{Feature: "plain.ffn_activation=geglu", Status: hfExportSupported, Reason: "Plain-block GeGLU FFN tails are mirrored with an explicit gate projection in the generated PyTorch template."},
		{Feature: "plain.ffn_activation=swiglu", Status: hfExportSupported, Reason: "Plain-block SwiGLU FFN tails are mirrored with an explicit gate projection in the generated PyTorch template."},
		{Feature: "plain.relative_attention=deberta_p2c_c2p", Status: hfExportSupported, Reason: "DeBERTa/GPT-BERT C2P/P2C relative bias uses log-bucketed q-k positions in the generated PyTorch template."},
		{Feature: "plain.relative_attention_parameterization=shared_qk_reuse", Status: hfExportSupported, Reason: "GPT-BERT-style shared relative embedding export reuses each block's Q/K projections in the generated PyTorch template."},
		{Feature: "plain.relative_attention_embedding_norm=layernorm", Status: hfExportSupported, Reason: "A model-level affine LayerNorm can be applied to the shared relative embedding before Q/K reuse."},
		{Feature: "layer_aggregation=dwa", Status: hfExportSupported, Reason: "Dense weighted aggregation is exported as one alpha vector per supported sublayer residual point."},
		{Feature: "mlm_head=bert", Status: hfExportSupported, Reason: "BERT-style masked LM transform head is exported with tied embedding output weight plus output bias."},
		{Feature: "swiglu", Status: hfExportSupported, Reason: "Bias-free SwiGLU FFN export is covered by native-vs-HF parity tests."},
		{Feature: "geglu", Status: hfExportSupported, Reason: "Bias-free GEGLU FFN export is covered by native-vs-HF parity tests."},
		{Feature: "mlp", Status: hfExportSupported, Reason: "Bias-free MLP export supports silu, gelu, relu, and leaky_relu_sq."},
		{Feature: "moe", Status: hfExportSupported, Reason: "Sequential linear-router top-k MoE export supports swiglu, geglu, and mlp experts."},
		{Feature: "hybrid_objective_eval", Status: hfExportSupported, Reason: "Hybrid configs export the causal next-token evaluation graph."},
		{Feature: "hgrn2", Status: hfExportGated, Reason: "Matrix-state scan export is intentionally gated until the PyTorch template has explicit recurrent-state parity coverage."},
		{Feature: "mlstm", Status: hfExportGated, Reason: "Stabilized matrix-memory scan export is intentionally gated until the PyTorch template has explicit recurrent-state parity coverage."},
		{Feature: "gated_deltanet", Status: hfExportGated, Reason: "Chunked delta-rule recurrence uses native scan semantics that are not yet mirrored in the HF template."},
		{Feature: "mamba", Status: hfExportGated, Reason: "Mamba-family selective scans and short-conv variants are not yet represented in the HF template."},
		{Feature: "mamba3-canonical", Status: hfExportGated, Reason: "Canonical Mamba-3 relies on specialized native scan semantics and CUDA/MLX execution paths."},
		{Feature: "retnet", Status: hfExportGated, Reason: "Retention recurrence export needs dedicated parity fixtures before it can be enabled."},
		{Feature: "rwkv", Status: hfExportGated, Reason: "RWKV recurrence export needs dedicated parity fixtures before it can be enabled."},
		{Feature: "custom", Status: hfExportUnsupported, Reason: "Arbitrary JSON custom blocks cannot be converted into a static HF template safely."},
		{Feature: "mlm_mntp_objectives", Status: hfExportTrainingOnly, Reason: "Masked objectives are training programs, not AutoModelForCausalLM inference graphs."},
	}
}

func hfExportBlockCapability(block BlockSpec) hfExportCapability {
	switch strings.ToLower(strings.TrimSpace(block.Type)) {
	case "plain":
		if hfNormalizePlainAttnPostNorm(block.AttnPostNorm) != "inherit" {
			return capabilityByFeature("plain.attn_post_norm")
		}
		if block.AttnValueGate {
			return capabilityByFeature("plain.attn_value_gate")
		}
		if block.AttnBias {
			return capabilityByFeature("plain.attn_bias")
		}
		if activation := hfPlainFFNActivation(block); activation == "geglu" || activation == "swiglu" {
			return capabilityByFeature("plain.ffn_activation=" + activation)
		}
		if hfRelativeAttentionUsesSharedQKReuse(block) {
			if hfRelativeAttentionEmbeddingNorm(block) == "layernorm" {
				return capabilityByFeature("plain.relative_attention_embedding_norm=layernorm")
			}
			return capabilityByFeature("plain.relative_attention_parameterization=shared_qk_reuse")
		}
		if relativeAttentionEnabledForHF(block) {
			return capabilityByFeature("plain.relative_attention=deberta_p2c_c2p")
		}
		if block.QKNorm {
			return capabilityByFeature("plain.qk_norm")
		}
		return capabilityByFeature("plain")
	case "swiglu", "geglu", "mlp", "moe", "hgrn2", "mlstm", "gated_deltanet", "mamba", "mamba3", "gated_linear_ssm", "mamba3-canonical", "retnet", "rwkv", "custom":
		feature := strings.ToLower(strings.TrimSpace(block.Type))
		if feature == "mamba3" || feature == "gated_linear_ssm" {
			feature = "mamba"
		}
		return capabilityByFeature(feature)
	default:
		return hfExportCapability{Feature: block.Type, Status: hfExportUnsupported, Reason: "Block type is not registered for HF export."}
	}
}

func capabilityByFeature(feature string) hfExportCapability {
	for _, capability := range hfExportCapabilities() {
		if capability.Feature == feature {
			return capability
		}
	}
	return hfExportCapability{Feature: feature, Status: hfExportUnsupported, Reason: "Feature is not registered for HF export."}
}

func relativeAttentionEnabledForHF(block BlockSpec) bool {
	return strings.EqualFold(strings.TrimSpace(block.RelativeAttention), "deberta_p2c_c2p")
}

func hfRelativeAttentionParameterization(block BlockSpec) string {
	value := strings.ToLower(strings.TrimSpace(block.RelativeAttentionParameterization))
	if value == "" {
		return "per_block_projections"
	}
	return value
}

func hfRelativeAttentionUsesSharedQKReuse(block BlockSpec) bool {
	return relativeAttentionEnabledForHF(block) && hfRelativeAttentionParameterization(block) == "shared_qk_reuse"
}

func hfRelativeAttentionEmbeddingNorm(block BlockSpec) string {
	value := strings.ToLower(strings.TrimSpace(block.RelativeAttentionEmbeddingNorm))
	switch value {
	case "", "none", "off", "disabled", "false":
		return "none"
	case "layernorm", "layer_norm", "ln":
		return "layernorm"
	default:
		return value
	}
}

func hfNormalizePlainAttnPostNorm(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "inherit":
		return "inherit"
	case "none", "off", "disabled", "false":
		return "none"
	case "after", "after_out", "after_outproj", "after_out_proj", "after_output_projection":
		return "after_outproj"
	case "before", "before_out", "before_outproj", "before_out_proj", "before_output_projection", "pre_outproj", "pre_out_proj":
		return "before_outproj"
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func hfEffectivePlainAttnPostNorm(block BlockSpec, normPlacement string) string {
	switch hfNormalizePlainAttnPostNorm(block.AttnPostNorm) {
	case "inherit":
		switch strings.ToLower(strings.TrimSpace(normPlacement)) {
		case "post", "sandwich":
			return "after_outproj"
		default:
			return "none"
		}
	case "none":
		return "none"
	case "after_outproj":
		return "after_outproj"
	case "before_outproj":
		return "before_outproj"
	default:
		return hfNormalizePlainAttnPostNorm(block.AttnPostNorm)
	}
}

func hfConfigUsesSharedRelativeAttention(cfg *ArchConfig) bool {
	if cfg == nil {
		return false
	}
	for _, block := range cfg.Blocks {
		if hfRelativeAttentionUsesSharedQKReuse(block) {
			return true
		}
	}
	return false
}

func hfConfigUsesSharedRelativeEmbeddingNorm(cfg *ArchConfig) bool {
	if cfg == nil {
		return false
	}
	for _, block := range cfg.Blocks {
		if hfRelativeAttentionUsesSharedQKReuse(block) && hfRelativeAttentionEmbeddingNorm(block) == "layernorm" {
			return true
		}
	}
	return false
}
