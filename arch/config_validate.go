package arch

import (
	"fmt"
	"math"
	"strings"
)

func validateParallelResidual(cfg *ArchConfig, source string) error {
	plan, err := newParallelResidualPlan(cfg.Blocks, cfg.ParallelResidual)
	if err != nil {
		return fmt.Errorf("config %q %w", source, err)
	}
	if plan.any && cfg.UNet {
		return fmt.Errorf("config %q cannot enable parallel_residual with unet", source)
	}
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return fmt.Errorf("config %q blocks: %w", source, err)
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return fmt.Errorf("config %q %w", source, err)
	}
	if err := validateResidualScaleInits(cfg, source); err != nil {
		return err
	}
	return nil
}

func validateResidualScaleInits(cfg *ArchConfig, source string) error {
	for i, block := range cfg.Blocks {
		if block.ResidualScaleInit == nil {
			continue
		}
		if math.IsNaN(*block.ResidualScaleInit) || math.IsInf(*block.ResidualScaleInit, 0) {
			return fmt.Errorf("config %q blocks[%d].residual_scale_init=%g must be finite", source, i, *block.ResidualScaleInit)
		}
		if !cfg.BlockScales {
			return fmt.Errorf("config %q blocks[%d].residual_scale_init requires block_scales=true", source, i)
		}
		switch blockTypeKey(block) {
		case "plain", "swiglu", "geglu", "moe", "gated_deltanet", "hgrn2", "s4d":
		default:
			return fmt.Errorf("config %q blocks[%d].residual_scale_init is not supported for type=%q", source, i, block.Type)
		}
	}
	return nil
}

func validateNormPolicy(cfg *ArchConfig, source string) error {
	if isDefaultNormConfig(cfg) {
		return nil
	}
	plan, planErr := newParallelResidualPlan(cfg.Blocks, cfg.ParallelResidual)
	if cfg.ParallelResidual || (planErr == nil && plan.any) {
		return fmt.Errorf("config %q non-default norm settings are not supported with parallel_residual in this release", source)
	}
	if cfg.UNet {
		return fmt.Errorf("config %q non-default norm settings are not supported with unet in this release", source)
	}
	for i, block := range cfg.Blocks {
		switch blockTypeKey(block) {
		case "plain", "swiglu", "geglu", "mlp", "s4d":
			// supported by the configurable normalization path.
		case "mamba3-canonical", "gated_deltanet":
			if cfg.EffectiveNormPlacement() != NormPlacementPre {
				return fmt.Errorf(
					"config %q norm_placement=%q is not supported with blocks[%d].type=%q; modern recurrent mixers support configurable outer pre-norm only",
					source, cfg.EffectiveNormPlacement(), i, block.Type,
				)
			}
			if cfg.FFNInternalNorm {
				return fmt.Errorf("config %q ffn_internal_norm is not supported with blocks[%d].type=%q", source, i, block.Type)
			}
		default:
			return fmt.Errorf("config %q non-default norm settings are not supported with blocks[%d].type=%q in this release", source, i, block.Type)
		}
	}
	if cfg.EffectiveNormPlacement() == NormPlacementPostResidual {
		for i, block := range cfg.Blocks {
			if blockTypeKey(block) != "s4d" {
				return fmt.Errorf("config %q norm_placement=\"post_residual\" currently supports s4d blocks only (blocks[%d].type=%q)", source, i, block.Type)
			}
		}
	}
	return nil
}

func validateBatchNormPolicy(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.EffectiveNormSpec().Type != NormTypeBatchNorm {
		return nil
	}
	if cfg.Training.EffectiveObjective() != ObjectiveClassification {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" supports training.objective=\"classification\" only in this release", source)
	}
	if cfg.EffectiveNormPlacement() != NormPlacementPre {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" requires norm_placement=\"pre\" in this release", source)
	}
	if cfg.Training.BatchTokens <= 1 {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" requires training.batch_tokens > 1", source)
	}
	if cfg.Training.SWAStart > 0 {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" does not support SWA until running-stat recalibration is available", source)
	}
	if len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" does not support recurrence in this release", source)
	}
	for i, block := range cfg.Blocks {
		if strings.TrimSpace(block.WeightGroup) != "" {
			return fmt.Errorf("config %q norm_type=\"batchnorm\" does not support blocks[%d].weight_group in this release", source, i)
		}
	}
	if cfg.Training.TTTSteps > 0 {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" does not support training-time test-time-training settings", source)
	}
	if mode := strings.ToLower(strings.TrimSpace(cfg.EffectiveEvalSpec().TTTMode)); mode != "" && mode != "none" {
		return fmt.Errorf("config %q norm_type=\"batchnorm\" does not support eval.ttt_mode=%q", source, cfg.EffectiveEvalSpec().TTTMode)
	}
	return nil
}

func validateRecurrence(cfg *ArchConfig, source string) error {
	if cfg.Recurrence == nil {
		return nil
	}
	if len(cfg.Recurrence) != len(cfg.Blocks) {
		return fmt.Errorf("config %q recurrence length=%d must match blocks length=%d", source, len(cfg.Recurrence), len(cfg.Blocks))
	}
	for i, ref := range cfg.Recurrence {
		if ref < 0 || ref >= len(cfg.Blocks) {
			return fmt.Errorf("config %q recurrence[%d]=%d out of range [0,%d)", source, i, ref, len(cfg.Blocks))
		}
		if ref > i {
			return fmt.Errorf("config %q recurrence[%d]=%d is a forward reference", source, i, ref)
		}
		if cfg.Blocks[i].Type != cfg.Blocks[ref].Type {
			return fmt.Errorf("config %q recurrence[%d]=%d type mismatch: blocks[%d].type=%q blocks[%d].type=%q", source, i, ref, i, cfg.Blocks[i].Type, ref, cfg.Blocks[ref].Type)
		}
		if ref != i && blockTypeKey(cfg.Blocks[i]) == "s4d" && !s4dSobolevSharedControlsEqual(cfg.Blocks[ref], cfg.Blocks[i]) {
			return fmt.Errorf("config %q recurrence[%d]=%d requires matching S4D Sobolev filter controls", source, i, ref)
		}
	}
	return nil
}

// validateRecurrencePhases and validateRecurrencePhaseOrder live in
// arch/config_recurrence_phases.go.

func validateWeightGroups(cfg *ArchConfig, source string) error {
	type weightGroupInfo struct {
		idx  int
		spec BlockSpec
	}

	groups := make(map[string]weightGroupInfo)
	for i, block := range cfg.Blocks {
		group := strings.TrimSpace(block.WeightGroup)
		cfg.Blocks[i].WeightGroup = group
		if group == "" {
			continue
		}

		prev, ok := groups[group]
		if !ok {
			groups[group] = weightGroupInfo{idx: i, spec: cfg.Blocks[i]}
			continue
		}

		if blockTypeKey(prev.spec) != blockTypeKey(block) {
			return fmt.Errorf("config %q blocks[%d] weight_group=%q type mismatch with blocks[%d] (%q vs %q)", source, i, group, prev.idx, block.Type, prev.spec.Type)
		}

		if prevHeads, ok := weightGroupHeadCount(prev.spec); ok {
			gotHeads, gotOK := weightGroupHeadCount(block)
			if !gotOK || gotHeads != prevHeads {
				return fmt.Errorf("config %q blocks[%d] weight_group=%q heads=%d must match blocks[%d] heads=%d", source, i, group, gotHeads, prev.idx, prevHeads)
			}
		}

		if err := validateWeightGroupLayout(cfg, prev.idx, prev.spec, i, block); err != nil {
			return fmt.Errorf("config %q %w", source, err)
		}
	}

	return nil
}

func weightGroupHeadCount(spec BlockSpec) (int, bool) {
	switch blockTypeKey(spec) {
	case "plain", "retnet", "perceiver", "bottleneck", "cross_attention", "gated_deltanet", "hgrn2", "mlstm", "ttt_mlp":
		return spec.Heads, true
	case "custom":
		if spec.Heads <= 0 {
			return 1, true
		}
		return spec.Heads, true
	default:
		return 0, false
	}
}

func validateWeightGroupLayout(cfg *ArchConfig, firstIdx int, first BlockSpec, curIdx int, cur BlockSpec) error {
	if blockTypeKey(first) == "s4d" && !s4dSobolevSharedControlsEqual(first, cur) {
		return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] Sobolev filter controls", curIdx, cur.WeightGroup, firstIdx)
	}
	firstShapes, err := blockWeightShapes(first, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix)
	if err != nil {
		return fmt.Errorf("blocks[%d] weight_group=%q references invalid weight layout: %w", firstIdx, first.WeightGroup, err)
	}
	curShapes, err := blockWeightShapes(cur, cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix)
	if err != nil {
		return fmt.Errorf("blocks[%d] weight_group=%q has invalid weight layout: %w", curIdx, cur.WeightGroup, err)
	}
	if len(firstShapes) != len(curShapes) {
		return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
	}
	for i := range firstShapes {
		if firstShapes[i].Name != curShapes[i].Name {
			return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
		}
		if len(firstShapes[i].Shape) != len(curShapes[i].Shape) {
			return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
		}
		for dim := range firstShapes[i].Shape {
			if firstShapes[i].Shape[dim] != curShapes[i].Shape[dim] {
				return fmt.Errorf("blocks[%d] weight_group=%q must match blocks[%d] weight layout", curIdx, cur.WeightGroup, firstIdx)
			}
		}
	}
	return nil
}

func s4dSobolevSharedControlsEqual(a, b BlockSpec) bool {
	if a.S4DSobolevFilterEnabled() != b.S4DSobolevFilterEnabled() {
		return false
	}
	if !a.S4DSobolevFilterEnabled() {
		return true
	}
	alo, ahi, abounded := S4DSobolevBounds(a)
	blo, bhi, bbounded := S4DSobolevBounds(b)
	return a.SobolevFilter.BetaInit == b.SobolevFilter.BetaInit &&
		effectiveS4DSobolevLearningRate(a) == effectiveS4DSobolevLearningRate(b) &&
		EffectiveS4DSobolevTrainable(a) == EffectiveS4DSobolevTrainable(b) &&
		EffectiveS4DSobolevWeightDecay(a) == EffectiveS4DSobolevWeightDecay(b) &&
		EffectiveS4DSobolevGranularity(a) == EffectiveS4DSobolevGranularity(b) &&
		abounded == bbounded && alo == blo && ahi == bhi
}

// validateBlockSpec checks that a single block spec has a valid type.
func validateBlockSpec(b BlockSpec, source, groupName string, idx int) error {
	if blockTypeKey(b) != "s4d" {
		if strings.TrimSpace(b.Init) != "" {
			return fmt.Errorf("config %q %s[%d] init is valid only for type=s4d", source, groupName, idx)
		}
		if strings.TrimSpace(b.OutputTransform) != "" {
			return fmt.Errorf("config %q %s[%d] output_transform is valid only for type=s4d", source, groupName, idx)
		}
		if b.NSSM != 0 || strings.TrimSpace(b.Discretization) != "" || b.TrainableB || b.StateLR != nil || b.TieDropout {
			return fmt.Errorf("config %q %s[%d] n_ssm, discretization, trainable_b, state_lr, and tie_dropout are valid only for type=s4d", source, groupName, idx)
		}
	}
	if b.Bidirectional && !supportsBidirectionalMixer(b) {
		return fmt.Errorf("config %q %s[%d] bidirectional is supported only for type=s4d, type=mamba3-canonical, or type=gated_deltanet", source, groupName, idx)
	}
	switch b.Type {
	case "mamba":
		return fmt.Errorf(
			"config %q %s[%d] type=%q is retired because it is not a reference Mamba implementation; "+
				"use type=%q only to load a legacy Mixlab checkpoint, or type=%q for canonical Mamba-3",
			source, groupName, idx, b.Type, "legacy_mamba", "mamba3-canonical",
		)
	case "plain", "swiglu", "geglu", "mlp", "moe", "legacy_mamba", "gated_linear_ssm", "mamba3", "mamba3-canonical", "s4d", "gated_deltanet", "hgrn2", "mlstm", "ttt_mlp", "rwkv", "retnet", "perceiver", "bottleneck", "cross_attention", "token_blend":
		// valid
	case "custom":
		return validateCustomBlockSpec(b, source, groupName, idx)
	default:
		if _, err := lookupBlock(b); err != nil {
			return fmt.Errorf("config %q %s[%d] has invalid type %q (not in registry)", source, groupName, idx, b.Type)
		}
	}
	if b.Type == "plain" && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=plain requires heads > 0", source, groupName, idx)
	}
	if blockTypeKey(b) == "ttt_mlp" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp requires heads > 0", source, groupName, idx)
		}
		if b.chunkSizeSet && b.ChunkSize <= 0 {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp has invalid chunk_size=%d (must be > 0 when set)", source, groupName, idx, b.ChunkSize)
		}
		if (b.innerHiddenMultSet && b.InnerHiddenMult <= 0) || math.IsNaN(b.InnerHiddenMult) || math.IsInf(b.InnerHiddenMult, 0) {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp has invalid inner_hidden_mult=%g (must be finite and > 0 when set)", source, groupName, idx, b.InnerHiddenMult)
		}
		if (b.innerLRBaseSet && b.InnerLRBase <= 0) || math.IsNaN(b.InnerLRBase) || math.IsInf(b.InnerLRBase, 0) {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp has invalid inner_lr_base=%g (must be finite and > 0 when set)", source, groupName, idx, b.InnerLRBase)
		}
		if (b.innerLRInitSet && b.InnerLRInit <= 0) || math.IsNaN(b.InnerLRInit) || math.IsInf(b.InnerLRInit, 0) {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp has invalid inner_lr_init=%g (must be finite and > 0 when set)", source, groupName, idx, b.InnerLRInit)
		}
		if effectiveTTTMLPInnerLRInit(b) > effectiveTTTMLPInnerLRBase(b) {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp requires inner_lr_init <= inner_lr_base", source, groupName, idx)
		}
		if b.InnerLRWarmupSteps != nil && *b.InnerLRWarmupSteps < 0 {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp has invalid inner_lr_warmup_steps=%d (must be >= 0)", source, groupName, idx, *b.InnerLRWarmupSteps)
		}
	}
	if b.Type == "plain" && b.QKGain < 0 {
		return fmt.Errorf("config %q %s[%d] type=plain has invalid qk_gain=%g (must be >= 0)", source, groupName, idx, b.QKGain)
	}
	if b.Type == "plain" && b.WindowSize < 0 {
		return fmt.Errorf("config %q %s[%d] type=plain has invalid window_size=%d (must be >= 0)", source, groupName, idx, b.WindowSize)
	}
	if b.Type == "plain" {
		switch normalizeRopeConvention(b.RopeConvention) {
		case RopeConventionAdjacentPair, RopeConventionHalfRotation:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid rope_convention=%q (must be \"adjacent_pair\" or \"half_rotation\")", source, groupName, idx, b.RopeConvention)
		}
		switch normalizeAttentionMask(b.AttentionMask) {
		case "", AttentionMaskCausal, AttentionMaskBidirectional, AttentionMaskNone:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid attention_mask=%q (must be \"causal\", \"bidirectional\", or \"none\")", source, groupName, idx, b.AttentionMask)
		}
		switch normalizeRelativeAttention(b.RelativeAttention) {
		case "", RelativeAttentionNone, RelativeAttentionDebertaP2CC2P:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid relative_attention=%q (must be \"deberta_p2c_c2p\" or \"none\")", source, groupName, idx, b.RelativeAttention)
		}
		switch normalizeRelativeAttentionParameterization(b.RelativeAttentionParameterization) {
		case RelativeAttentionParamPerBlockProjections, RelativeAttentionParamSharedQKReuse:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid relative_attention_parameterization=%q (must be \"per_block_projections\" or \"shared_qk_reuse\")", source, groupName, idx, b.RelativeAttentionParameterization)
		}
		switch normalizeRelativeAttentionEmbeddingNorm(b.RelativeAttentionEmbeddingNorm) {
		case RelativeAttentionEmbeddingNormNone, RelativeAttentionEmbeddingNormLayerNorm:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid relative_attention_embedding_norm=%q (must be \"none\" or \"layernorm\")", source, groupName, idx, b.RelativeAttentionEmbeddingNorm)
		}
		switch normalizePlainAttnPostNorm(b.AttnPostNorm) {
		case PlainAttnPostNormInherit, PlainAttnPostNormNone, PlainAttnPostNormAfterOutProj, PlainAttnPostNormBeforeOutProj:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid attn_post_norm=%q (must be \"inherit\", \"none\", \"after_outproj\", or \"before_outproj\")", source, groupName, idx, b.AttnPostNorm)
		}
		switch normalizePlainFFNActivation(b.FFNActivation) {
		case PlainFFNActivationSiLU, PlainFFNActivationGEGLU, PlainFFNActivationSwiGLU, PlainFFNActivationGELU, PlainFFNActivationGELUNew:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid ffn_activation=%q (must be \"silu\", \"gelu\", \"gelu_new\", \"geglu\", or \"swiglu\")", source, groupName, idx, b.FFNActivation)
		}
		if normalizeRelativeAttentionParameterization(b.RelativeAttentionParameterization) == RelativeAttentionParamSharedQKReuse && !relativeAttentionEnabled(b) {
			return fmt.Errorf("config %q %s[%d] type=plain relative_attention_parameterization=\"shared_qk_reuse\" requires relative_attention=\"deberta_p2c_c2p\"", source, groupName, idx)
		}
		if normalizeRelativeAttentionEmbeddingNorm(b.RelativeAttentionEmbeddingNorm) != RelativeAttentionEmbeddingNormNone && !relativeAttentionUsesSharedQKReuse(b) {
			return fmt.Errorf("config %q %s[%d] type=plain relative_attention_embedding_norm requires relative_attention_parameterization=\"shared_qk_reuse\"", source, groupName, idx)
		}
		if b.AttnValueGate && b.KVSource > 0 {
			return fmt.Errorf("config %q %s[%d] type=plain cannot combine attn_value_gate with kv_source", source, groupName, idx)
		}
		if b.RelativeAttentionWindow < 0 {
			return fmt.Errorf("config %q %s[%d] type=plain has invalid relative_attention_window=%d (must be >= 0)", source, groupName, idx, b.RelativeAttentionWindow)
		}
		if relativeAttentionEnabled(b) {
			if b.RopeDims != 0 {
				return fmt.Errorf("config %q %s[%d] type=plain cannot combine relative_attention with rope_dims", source, groupName, idx)
			}
			if strings.TrimSpace(b.RopeConvention) != "" {
				return fmt.Errorf("config %q %s[%d] type=plain cannot combine relative_attention with rope_convention", source, groupName, idx)
			}
			if b.KVSource > 0 {
				return fmt.Errorf("config %q %s[%d] type=plain cannot combine relative_attention with kv_source", source, groupName, idx)
			}
		}
	}
	if b.Type == "plain" && b.KVHeads != 0 {
		if b.KVHeads < 0 {
			return fmt.Errorf("config %q %s[%d] type=plain has invalid kv_heads=%d (must be > 0 when set)", source, groupName, idx, b.KVHeads)
		}
		if b.Heads%b.KVHeads != 0 {
			return fmt.Errorf("config %q %s[%d] type=plain requires heads %% kv_heads == 0 (got heads=%d kv_heads=%d)", source, groupName, idx, b.Heads, b.KVHeads)
		}
	}
	if b.Type == "retnet" && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=retnet requires heads > 0", source, groupName, idx)
	}
	if b.Type == "gated_deltanet" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=gated_deltanet requires heads > 0", source, groupName, idx)
		}
		if b.DK <= 0 {
			return fmt.Errorf("config %q %s[%d] type=gated_deltanet requires d_k > 0", source, groupName, idx)
		}
		dv := b.DV
		if dv <= 0 {
			dv = 2 * b.DK
		}
		if dv <= 0 {
			return fmt.Errorf("config %q %s[%d] type=gated_deltanet has invalid d_v=%d", source, groupName, idx, b.DV)
		}
		if effectiveKVShare(b) && dv < b.DK {
			return fmt.Errorf("config %q %s[%d] type=gated_deltanet with kv_share=true requires d_v >= d_k (got d_v=%d d_k=%d)", source, groupName, idx, dv, b.DK)
		}
		if b.ScanChunkSize != nil && *b.ScanChunkSize < 0 {
			return fmt.Errorf("config %q %s[%d] type=gated_deltanet has invalid scan_chunk_size=%d (must be >= 0)", source, groupName, idx, *b.ScanChunkSize)
		}
	}
	if b.Type == "hgrn2" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=hgrn2 requires heads > 0", source, groupName, idx)
		}
		if b.DState < 0 {
			return fmt.Errorf("config %q %s[%d] type=hgrn2 has invalid d_state=%d (must be > 0 when set)", source, groupName, idx, b.DState)
		}
	}
	if b.Type == "mlstm" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=mlstm requires heads > 0", source, groupName, idx)
		}
		if b.DK <= 0 {
			return fmt.Errorf("config %q %s[%d] type=mlstm requires d_k > 0", source, groupName, idx)
		}
		if b.DV <= 0 {
			return fmt.Errorf("config %q %s[%d] type=mlstm requires d_v > 0", source, groupName, idx)
		}
	}
	if b.Type == "mamba3-canonical" {
		if b.StateSize < 0 || b.NGroups < 0 || b.DTRank < 0 || b.ConvKernel < 0 {
			return fmt.Errorf("config %q %s[%d] type=mamba3-canonical has negative dimension field", source, groupName, idx)
		}
		if b.ScanChunkSize != nil && *b.ScanChunkSize < 0 {
			return fmt.Errorf("config %q %s[%d] type=mamba3-canonical has invalid scan_chunk_size=%d (must be >= 0)", source, groupName, idx, *b.ScanChunkSize)
		}
		if b.StateSize > 0 && b.StateSize%2 != 0 {
			return fmt.Errorf("config %q %s[%d] type=mamba3-canonical requires even state_size for complex state pairs", source, groupName, idx)
		}
		if b.DTMin < 0 || b.DTMax < 0 || (b.DTMin > 0 && b.DTMax > 0 && b.DTMax <= b.DTMin) {
			return fmt.Errorf("config %q %s[%d] type=mamba3-canonical requires 0 < dt_min < dt_max when set", source, groupName, idx)
		}
	}
	if blockTypeKey(b) == "s4d" {
		stateSize := effectiveS4DStateSize(b)
		if stateSize <= 0 || stateSize%2 != 0 {
			return fmt.Errorf("config %q %s[%d] type=s4d requires a positive even state_size (got %d)", source, groupName, idx, stateSize)
		}
		switch effectiveS4DInit(b) {
		case S4DInitLin:
			// Reference-locked v1 initialization.
		default:
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid init=%q (v1 supports %q)", source, groupName, idx, b.Init, S4DInitLin)
		}
		dtMin, dtMax := effectiveS4DDTRange(b)
		if !(dtMin > 0) || !(dtMax > dtMin) || math.IsNaN(dtMin) || math.IsNaN(dtMax) ||
			math.IsInf(dtMin, 0) || math.IsInf(dtMax, 0) {
			return fmt.Errorf("config %q %s[%d] type=s4d requires 0 < dt_min < dt_max", source, groupName, idx)
		}
		if b.NSSM < 0 {
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid n_ssm=%d (must be > 0 when set)", source, groupName, idx, b.NSSM)
		}
		switch effectiveS4DDiscretization(b) {
		case S4DDiscretizationZOH, S4DDiscretizationBilinear:
		default:
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid discretization=%q (must be \"zoh\" or \"bilinear\")", source, groupName, idx, b.Discretization)
		}
		if b.StateLR != nil && (*b.StateLR <= 0 || math.IsNaN(*b.StateLR) || math.IsInf(*b.StateLR, 0)) {
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid state_lr=%g (must be finite and > 0)", source, groupName, idx, *b.StateLR)
		}
		freqScale := effectiveS4DFreqScale(b)
		if !(freqScale > 0) || math.IsNaN(freqScale) || math.IsInf(freqScale, 0) {
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid freq_scale=%g (must be finite and > 0)", source, groupName, idx, freqScale)
		}
		if b.SobolevFilter != nil {
			if math.IsNaN(b.SobolevFilter.BetaInit) || math.IsInf(b.SobolevFilter.BetaInit, 0) {
				return fmt.Errorf("config %q %s[%d] type=s4d has invalid sobolev_filter.beta_init=%g (must be finite)", source, groupName, idx, b.SobolevFilter.BetaInit)
			}
			if b.S4DSobolevFilterEnabled() {
				lr := effectiveS4DSobolevLearningRate(b)
				if math.IsNaN(lr) || math.IsInf(lr, 0) || lr < 0 || (EffectiveS4DSobolevTrainable(b) && lr == 0) {
					return fmt.Errorf("config %q %s[%d] type=s4d has invalid sobolev_filter.learning_rate=%g (must be finite and > 0 when trainable, or >= 0 when frozen)", source, groupName, idx, lr)
				}
				decay := EffectiveS4DSobolevWeightDecay(b)
				if decay < 0 || math.IsNaN(decay) || math.IsInf(decay, 0) {
					return fmt.Errorf("config %q %s[%d] type=s4d has invalid sobolev_filter.weight_decay=%g (must be finite and >= 0)", source, groupName, idx, decay)
				}
				switch EffectiveS4DSobolevGranularity(b) {
				case S4DSobolevGranularityChannel, S4DSobolevGranularityLayer:
				default:
					return fmt.Errorf("config %q %s[%d] type=s4d has invalid sobolev_filter.granularity=%q (must be \"channel\" or \"layer\")", source, groupName, idx, b.SobolevFilter.Granularity)
				}
				if len(b.SobolevFilter.Bounds) != 0 && len(b.SobolevFilter.Bounds) != 2 {
					return fmt.Errorf("config %q %s[%d] type=s4d requires sobolev_filter.bounds to contain exactly [min,max]", source, groupName, idx)
				}
				if lo, hi, bounded := S4DSobolevBounds(b); bounded {
					if math.IsNaN(lo) || math.IsNaN(hi) || math.IsInf(lo, 0) || math.IsInf(hi, 0) || !(lo < hi) {
						return fmt.Errorf("config %q %s[%d] type=s4d requires sobolev_filter.bounds=[min,max] with finite min < max", source, groupName, idx)
					}
					if !(b.SobolevFilter.BetaInit > lo && b.SobolevFilter.BetaInit < hi) {
						return fmt.Errorf("config %q %s[%d] type=s4d requires sobolev_filter.beta_init=%g strictly inside bounds [%g,%g]", source, groupName, idx, b.SobolevFilter.BetaInit, lo, hi)
					}
				}
			}
		}
		switch effectiveS4DOutputTransform(b) {
		case S4DOutputTransformNone, S4DOutputTransformGLU:
		default:
			return fmt.Errorf("config %q %s[%d] type=s4d has invalid output_transform=%q (must be \"none\" or \"glu\")", source, groupName, idx, b.OutputTransform)
		}
	}
	if (b.Type == "perceiver" || b.Type == "bottleneck") && b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] type=%s requires heads > 0", source, groupName, idx, b.Type)
	}
	if b.Type == "cross_attention" {
		if b.Heads <= 0 {
			return fmt.Errorf("config %q %s[%d] type=cross_attention requires heads > 0", source, groupName, idx)
		}
		if b.SourceStream == "" {
			return fmt.Errorf("config %q %s[%d] type=cross_attention requires source_stream", source, groupName, idx)
		}
	}
	if blockTypeKey(b) == "mlp" {
		switch strings.ToLower(strings.TrimSpace(b.Activation)) {
		case "", "silu", "gelu", "relu", "leaky_relu_sq":
			// valid
		default:
			return fmt.Errorf("config %q %s[%d] type=mlp has invalid activation %q", source, groupName, idx, b.Activation)
		}
		if b.LeakySlope < 0 {
			return fmt.Errorf("config %q %s[%d] type=mlp has invalid leaky_slope=%g (must be >= 0)", source, groupName, idx, b.LeakySlope)
		}
	}
	if blockTypeKey(b) == "moe" {
		if b.NumExperts <= 0 {
			return fmt.Errorf("config %q %s[%d] type=moe requires num_experts > 0", source, groupName, idx)
		}
		topK := effectiveMoETopK(b)
		if topK < 1 || topK > b.NumExperts {
			return fmt.Errorf("config %q %s[%d] type=moe has invalid top_k=%d (must be in [1,num_experts=%d])", source, groupName, idx, topK, b.NumExperts)
		}
		if moeRouter(b) != "linear" {
			return fmt.Errorf("config %q %s[%d] type=moe has invalid router=%q (v1 supports \"linear\")", source, groupName, idx, b.Router)
		}
		if effectiveMoELoadBalanceLossWeight(b) < 0 {
			return fmt.Errorf("config %q %s[%d] type=moe has invalid load_balance_loss_weight=%g (must be >= 0)", source, groupName, idx, effectiveMoELoadBalanceLossWeight(b))
		}
		expert := effectiveMoEExpertBlock(b)
		if expert.WeightGroup != "" {
			return fmt.Errorf("config %q %s[%d] type=moe expert_block cannot set weight_group", source, groupName, idx)
		}
		if expert.ParallelResidual != nil {
			return fmt.Errorf("config %q %s[%d] type=moe expert_block cannot set parallel_residual", source, groupName, idx)
		}
		switch blockTypeKey(expert) {
		case "swiglu", "geglu":
			// valid
		case "mlp":
			switch strings.ToLower(strings.TrimSpace(expert.Activation)) {
			case "", "silu", "gelu", "relu", "leaky_relu_sq":
				// valid
			default:
				return fmt.Errorf("config %q %s[%d] type=moe expert_block has invalid activation %q", source, groupName, idx, expert.Activation)
			}
			if expert.LeakySlope < 0 {
				return fmt.Errorf("config %q %s[%d] type=moe expert_block has invalid leaky_slope=%g (must be >= 0)", source, groupName, idx, expert.LeakySlope)
			}
		default:
			return fmt.Errorf("config %q %s[%d] type=moe expert_block.type must be swiglu, geglu, or mlp (got %q)", source, groupName, idx, expert.Type)
		}
	}
	return nil
}

func validateSharedRelativeAttention(cfg *ArchConfig, source string) error {
	if _, err := newSharedRelativeAttentionPlan(cfg.Blocks); err != nil {
		if _, ok := err.(*sharedRelativeWindowMismatchError); ok {
			return fmt.Errorf("config %q blocks with relative_attention_parameterization=\"shared_qk_reuse\" must use the same effective relative_attention_window", source)
		}
		return fmt.Errorf("config %q %w", source, err)
	}
	return nil
}

func validateKVSources(cfg *ArchConfig, source string) error {
	for i, b := range cfg.Blocks {
		if blockTypeKey(b) != "plain" || b.KVSource <= 0 {
			continue
		}
		srcIdx := b.KVSource - 1
		if srcIdx < 0 || srcIdx >= len(cfg.Blocks) {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (must reference an earlier block)", source, i, b.KVSource)
		}
		if srcIdx >= i {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (must reference an earlier block)", source, i, b.KVSource)
		}
		src := cfg.Blocks[srcIdx]
		if blockTypeKey(src) != "plain" {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (blocks[%d] is type=%q, want plain)", source, i, b.KVSource, srcIdx, src.Type)
		}
		if relativeAttentionEnabled(src) {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (source block uses relative_attention)", source, i, b.KVSource)
		}
		if src.DifferentialAttention {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d (source block uses differential_attention)", source, i, b.KVSource)
		}

		wantKVHeads, err := normalizePlainKVHeads(b.Heads, b.KVHeads)
		if err != nil {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid kv_source=%d: %w", source, i, b.KVSource, err)
		}
		gotKVHeads, err := normalizePlainKVHeads(src.Heads, src.KVHeads)
		if err != nil {
			return fmt.Errorf("config %q blocks[%d] kv_source=%d references invalid source block: %w", source, i, b.KVSource, err)
		}
		if src.Heads != b.Heads {
			return fmt.Errorf("config %q blocks[%d] type=plain has incompatible kv_source=%d (heads=%d, source heads=%d)", source, i, b.KVSource, b.Heads, src.Heads)
		}
		if gotKVHeads != wantKVHeads {
			return fmt.Errorf("config %q blocks[%d] type=plain has incompatible kv_source=%d (kv_heads=%d, source kv_heads=%d)", source, i, b.KVSource, wantKVHeads, gotKVHeads)
		}
	}
	return nil
}

func validateBlockRopeDims(b BlockSpec, modelDim int, source, groupName string, idx int) error {
	if blockTypeKey(b) == "plain" && relativeAttentionEnabled(b) {
		if b.Heads <= 0 {
			return nil
		}
		if modelDim%b.Heads != 0 {
			return fmt.Errorf("config %q %s[%d] type=plain with relative_attention requires model_dim=%d divisible by heads=%d", source, groupName, idx, modelDim, b.Heads)
		}
	}
	if b.RopeDims == 0 {
		return nil
	}
	if b.RopeDims < 0 {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be > 0 when set)", source, groupName, idx, b.RopeDims)
	}
	if b.RopeDims%2 != 0 {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be even)", source, groupName, idx, b.RopeDims)
	}
	if b.Heads <= 0 {
		return fmt.Errorf("config %q %s[%d] has rope_dims=%d but heads must be > 0", source, groupName, idx, b.RopeDims)
	}
	if modelDim%b.Heads != 0 {
		return fmt.Errorf("config %q %s[%d] has rope_dims=%d but model_dim=%d is not divisible by heads=%d", source, groupName, idx, b.RopeDims, modelDim, b.Heads)
	}
	headDim := modelDim / b.Heads
	if b.RopeDims > headDim {
		return fmt.Errorf("config %q %s[%d] has invalid rope_dims=%d (must be <= head_dim=%d)", source, groupName, idx, b.RopeDims, headDim)
	}
	return nil
}

func validateRecurrentMixerDims(b BlockSpec, modelDim int, source, groupName string, idx int) error {
	switch blockTypeKey(b) {
	case "s4d":
		nSSM := effectiveS4DNSSM(b, modelDim)
		if nSSM <= 0 || nSSM > modelDim || modelDim%nSSM != 0 {
			return fmt.Errorf("config %q %s[%d] type=s4d requires n_ssm to divide model_dim (got n_ssm=%d model_dim=%d)", source, groupName, idx, nSSM, modelDim)
		}
	case "hgrn2":
		if b.Heads <= 0 {
			return nil
		}
		if modelDim%b.Heads != 0 {
			return fmt.Errorf("config %q %s[%d] type=hgrn2 requires model_dim=%d divisible by heads=%d", source, groupName, idx, modelDim, b.Heads)
		}
	case "mlstm":
		// mLSTM projects to heads*d_k and heads*d_v, so model_dim does not need
		// to divide evenly by heads.
	case "ttt_mlp":
		if b.Heads <= 0 {
			return nil
		}
		if modelDim%b.Heads != 0 {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp requires model_dim=%d divisible by heads=%d", source, groupName, idx, modelDim, b.Heads)
		}
		if (modelDim/b.Heads)%2 != 0 {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp requires an even head_dim for chunk-relative RoPE", source, groupName, idx)
		}
		if _, err := effectiveTTTMLPInnerHiddenDim(b, modelDim); err != nil {
			return fmt.Errorf("config %q %s[%d] type=ttt_mlp: %w", source, groupName, idx, err)
		}
	}
	return nil
}

// validateCustomBlockSpec validates a custom block's weights and ops.
func validateCustomBlockSpec(b BlockSpec, source, groupName string, idx int) error {
	if b.Name == "" {
		return fmt.Errorf("config %q %s[%d] type=custom requires a name", source, groupName, idx)
	}
	if len(b.Weights) == 0 {
		return fmt.Errorf("config %q %s[%d] custom block %q must declare at least one weight", source, groupName, idx, b.Name)
	}
	if len(b.Ops) == 0 {
		return fmt.Errorf("config %q %s[%d] custom block %q must declare at least one op", source, groupName, idx, b.Name)
	}
	for wi, w := range b.Weights {
		if w.Name == "" {
			return fmt.Errorf("config %q %s[%d] custom block %q weight[%d] missing name", source, groupName, idx, b.Name, wi)
		}
		if len(w.Shape) == 0 {
			return fmt.Errorf("config %q %s[%d] custom block %q weight %q missing shape", source, groupName, idx, b.Name, w.Name)
		}
	}
	for oi, op := range b.Ops {
		if op.Op == "" {
			return fmt.Errorf("config %q %s[%d] custom block %q op[%d] missing op name", source, groupName, idx, b.Name, oi)
		}
		if op.Output == "" && len(op.Outputs) == 0 {
			return fmt.Errorf("config %q %s[%d] custom block %q op[%d] missing output(s)", source, groupName, idx, b.Name, oi)
		}
	}
	return nil
}
