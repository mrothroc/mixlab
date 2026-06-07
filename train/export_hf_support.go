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
		{Feature: "plain", Status: hfExportSupported, Reason: "Core attention export with RoPE, GQA, qk_norm, qk_gain, masks, and causal windowing."},
		{Feature: "plain.qk_norm", Status: hfExportSupported, Reason: "Learned Q/K RMSNorm scales are mirrored in the generated PyTorch template."},
		{Feature: "plain.relative_attention=deberta_p2c_c2p", Status: hfExportSupported, Reason: "DeBERTa C2P/P2C relative bias is mirrored in the generated PyTorch template."},
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
