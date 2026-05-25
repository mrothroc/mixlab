package arch

import (
	"fmt"
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
	case "plain", "retnet", "perceiver", "bottleneck", "cross_attention", "gated_deltanet", "hgrn2", "mlstm":
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

// validateBlockSpec checks that a single block spec has a valid type.
func validateBlockSpec(b BlockSpec, source, groupName string, idx int) error {
	switch b.Type {
	case "plain", "swiglu", "geglu", "mlp", "mamba", "gated_linear_ssm", "mamba3", "mamba3-canonical", "gated_deltanet", "hgrn2", "mlstm", "rwkv", "retnet", "perceiver", "bottleneck", "cross_attention", "token_blend":
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
	if b.Type == "plain" && b.QKGain < 0 {
		return fmt.Errorf("config %q %s[%d] type=plain has invalid qk_gain=%g (must be >= 0)", source, groupName, idx, b.QKGain)
	}
	if b.Type == "plain" && b.WindowSize < 0 {
		return fmt.Errorf("config %q %s[%d] type=plain has invalid window_size=%d (must be >= 0)", source, groupName, idx, b.WindowSize)
	}
	if b.Type == "plain" {
		switch normalizeAttentionMask(b.AttentionMask) {
		case "", AttentionMaskCausal, AttentionMaskBidirectional, AttentionMaskNone:
		default:
			return fmt.Errorf("config %q %s[%d] type=plain has invalid attention_mask=%q (must be \"causal\", \"bidirectional\", or \"none\")", source, groupName, idx, b.AttentionMask)
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
