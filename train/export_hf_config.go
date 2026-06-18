package train

import "strings"

func writeHFConfig(path string, cfg *ArchConfig, specials hfTokenizerSpecials) error {
	blocks := hfBlockEntries(cfg, false)
	maskedBlocks := []map[string]any(nil)
	architectures := []string{"MixlabForCausalLM"}
	autoMap := map[string]string{
		"AutoConfig":           "configuration_mixlab.MixlabConfig",
		"AutoModel":            "modeling_mixlab.MixlabModel",
		"AutoModelForCausalLM": "modeling_mixlab.MixlabForCausalLM",
	}
	if hfExportSupportsMaskedLM(cfg) {
		maskedBlocks = hfBlockEntries(cfg, true)
		architectures = append(architectures, "MixlabForMaskedLM")
		autoMap["AutoModelForMaskedLM"] = "modeling_mixlab.MixlabForMaskedLM"
	}
	doc := hfConfigJSON{
		ModelType:             "mixlab",
		Architectures:         architectures,
		AutoMap:               autoMap,
		Name:                  cfg.Name,
		ModelDim:              cfg.ModelDim,
		HiddenSize:            cfg.ModelDim,
		VocabSize:             cfg.VocabSize,
		SeqLen:                cfg.SeqLen,
		MaxPositionEmbeddings: cfg.SeqLen,
		MLPMult:               cfg.EffectiveMLPMult(),
		NormType:              cfg.EffectiveNormSpec().Type,
		NormEps:               cfg.EffectiveNormSpec().Eps,
		NormAffine:            cfg.EffectiveNormSpec().Affine,
		NormPlacement:         cfg.EffectiveNormPlacement(),
		FFNInternalNorm:       cfg.FFNInternalNorm,
		LogitSoftcap:          cfg.LogitSoftcap,
		MLMHead:               hfExportMLMHead(cfg),
		HiddenDropout:         cfg.EffectiveHiddenDropout(),
		CharVocabSize:         cfg.CharVocabSize,
		CharDim:               cfg.EffectiveCharDim(),
		CharMaxPerToken:       cfg.EffectiveCharMaxPerToken(),
		CharFeaturesFile:      charFeaturesFileForHFConfig(cfg),
		BigramVocabSize:       cfg.BigramVocabSize,
		BigramDim:             cfg.EffectiveBigramDim(),
		TrigramVocabSize:      cfg.TrigramVocabSize,
		TrigramDim:            cfg.EffectiveTrigramDim(),
		PadTokenID:            specialTokenIDPtr(specials.Pad),
		EOSTokenID:            specialTokenIDPtr(specials.EOS),
		BOSTokenID:            specialTokenIDPtr(specials.BOS),
		UNKTokenID:            specialTokenIDPtr(specials.UNK),
		Blocks:                blocks,
		MaskedBlocks:          maskedBlocks,
		Mixlab: map[string]any{
			"format":            "mixlab_hf_export_v1",
			"source":            "mixlab",
			"weight_map":        "weight_map.json",
			"requires_trust":    "trust_remote_code=True loads repository-provided Python modeling code",
			"supported_blocks":  []string{"plain", "plain.attn_bias", "plain.attn_value_gate", "plain.attn_post_norm", "plain.ffn_activation=geglu", "plain.ffn_activation=swiglu", "plain.qk_norm", "plain.xsa", "plain.sparse_attn_gate", "plain.relative_attention=deberta_p2c_c2p", "plain.relative_attention_parameterization=shared_qk_reuse", "plain.relative_attention_embedding_norm=layernorm", "mlm_head=bert", "swiglu", "geglu", "mlp", "moe"},
			"unsupported_fails": true,
		},
	}
	return writeJSONFile(path, doc)
}

func hfExportMLMHead(cfg *ArchConfig) string {
	if cfg == nil || cfg.EffectiveMLMHead() != "bert" {
		return ""
	}
	return "bert"
}

func hfBlockEntries(cfg *ArchConfig, masked bool) []map[string]any {
	blocks := make([]map[string]any, 0, len(cfg.Blocks))
	for _, block := range cfg.Blocks {
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			entry := map[string]any{
				"type":  "plain",
				"heads": block.Heads,
			}
			if block.KVHeads > 0 && block.KVHeads != block.Heads {
				entry["kv_heads"] = block.KVHeads
			}
			if block.RopeDims > 0 {
				entry["rope_dims"] = block.RopeDims
			}
			if normalizeHFRopeConvention(block.RopeConvention) == "half_rotation" {
				entry["rope_convention"] = "half_rotation"
			}
			if block.QKGain > 0 {
				entry["qk_gain"] = block.QKGain
			}
			if block.QKNorm {
				entry["qk_norm"] = true
			}
			if block.AttnBias {
				entry["attn_bias"] = true
			}
			if block.AttnValueGate {
				entry["attn_value_gate"] = true
			}
			if mode := hfNormalizePlainAttnPostNorm(block.AttnPostNorm); mode != "inherit" {
				entry["attn_post_norm"] = mode
			}
			if block.XSA {
				entry["xsa"] = true
			}
			if block.SparseAttnGate {
				entry["sparse_attn_gate"] = true
			}
			if block.WindowSize > 0 && !masked {
				entry["window_size"] = block.WindowSize
			}
			if relativeAttentionEnabledForHF(block) {
				entry["relative_attention"] = "deberta_p2c_c2p"
				entry["relative_attention_window"] = effectiveHFRelativeAttentionWindow(block)
				if hfRelativeAttentionUsesSharedQKReuse(block) {
					entry["relative_attention_parameterization"] = "shared_qk_reuse"
					if hfRelativeAttentionEmbeddingNorm(block) == "layernorm" {
						entry["relative_attention_embedding_norm"] = "layernorm"
					}
				}
			}
			if activation := hfPlainFFNActivation(block); activation != "silu" {
				entry["ffn_activation"] = activation
			}
			mask := hfExportAttentionMask(cfg, block, masked)
			if mask != "" {
				entry["attention_mask"] = mask
			}
			blocks = append(blocks, entry)
		case "swiglu":
			blocks = append(blocks, map[string]any{"type": "swiglu"})
		case "geglu":
			blocks = append(blocks, map[string]any{"type": "geglu"})
		case "mlp":
			entry := map[string]any{"type": "mlp"}
			if strings.TrimSpace(block.Activation) != "" {
				entry["activation"] = strings.ToLower(strings.TrimSpace(block.Activation))
			}
			if block.LeakySlope != 0 {
				entry["leaky_slope"] = block.LeakySlope
			}
			blocks = append(blocks, entry)
		case "moe":
			entry := map[string]any{
				"type":        "moe",
				"num_experts": block.NumExperts,
				"top_k":       effectiveHFMoETopK(block),
				"router":      "linear",
			}
			expert := BlockSpec{Type: "swiglu"}
			if block.ExpertBlock != nil {
				expert = *block.ExpertBlock
			}
			expertEntry := map[string]any{"type": strings.ToLower(strings.TrimSpace(expert.Type))}
			if expertEntry["type"] == "" {
				expertEntry["type"] = "swiglu"
			}
			if expertEntry["type"] == "mlp" {
				if strings.TrimSpace(expert.Activation) != "" {
					expertEntry["activation"] = strings.ToLower(strings.TrimSpace(expert.Activation))
				}
				if expert.LeakySlope != 0 {
					expertEntry["leaky_slope"] = expert.LeakySlope
				}
			}
			entry["expert_block"] = expertEntry
			blocks = append(blocks, entry)
		}
	}
	return blocks
}

func hfExportSupportsMaskedLM(cfg *ArchConfig) bool {
	if cfg == nil {
		return false
	}
	switch cfg.Training.EffectiveObjective() {
	case "mlm", "mntp":
		return true
	case "hybrid":
		return cfg.Training.HybridCLMFraction < 1
	default:
		return false
	}
}

func normalizeHFRopeConvention(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "half", "half_rotation", "half-rotation":
		return "half_rotation"
	default:
		return "adjacent_pair"
	}
}

func hfExportAttentionMask(cfg *ArchConfig, block BlockSpec, masked bool) string {
	if masked {
		return "bidirectional"
	}
	if cfg != nil {
		switch cfg.Training.EffectiveObjective() {
		case "hybrid", "mlm", "mntp":
			return "causal"
		}
	}
	mask := strings.ToLower(strings.TrimSpace(block.AttentionMask))
	if mask == "" {
		return "causal"
	}
	return mask
}

func effectiveHFRelativeAttentionWindow(block BlockSpec) int {
	if block.RelativeAttentionWindow > 0 {
		return block.RelativeAttentionWindow
	}
	return 128
}

func effectiveHFMoETopK(block BlockSpec) int {
	if block.TopK > 0 {
		return block.TopK
	}
	if block.NumExperts <= 1 {
		return 1
	}
	return 2
}

func charFeaturesFileForHFConfig(cfg *ArchConfig) string {
	if cfg != nil && cfg.CharVocabSize > 0 {
		return charFeaturesFilename
	}
	return ""
}
