package arch

import (
	"fmt"
	"math"
)

func tttMLPWeightCount(_ BlockSpec, blockScales, _ bool) (int, error) {
	total := 20
	if blockScales {
		total++
	}
	return total, nil
}

func tttMLPWeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	return tttMLPWeightShapesWithOptions(spec, D, false)
}

func tttMLPWeightShapesWithOptions(spec BlockSpec, D int, blockScales bool) ([]WeightMeta, error) {
	if spec.Heads <= 0 || D%spec.Heads != 0 {
		return nil, fmt.Errorf("ttt_mlp requires model_dim divisible by heads")
	}
	headDim := D / spec.Heads
	if headDim%2 != 0 {
		return nil, fmt.Errorf("ttt_mlp requires an even head_dim for chunk-relative RoPE")
	}
	hidden, err := effectiveTTTMLPInnerHiddenDim(spec, D)
	if err != nil {
		return nil, err
	}
	chunk := effectiveTTTMLPChunkSize(spec)
	metas := []WeightMeta{
		{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "w_qk", Shape: []int{D, D}, InitMode: "ttt_normal_0_02"},
		{Name: "q_conv", Shape: []int{D, defaultTTTMLPConvKernel}, InitMode: "ttt_conv_uniform_4"},
		{Name: "q_conv_bias", Shape: []int{D}, InitMode: "ttt_conv_uniform_4"},
		{Name: "k_conv", Shape: []int{D, defaultTTTMLPConvKernel}, InitMode: "ttt_conv_uniform_4"},
		{Name: "k_conv_bias", Shape: []int{D}, InitMode: "ttt_conv_uniform_4"},
		{Name: "w_v", Shape: []int{D, D}, InitMode: "ttt_normal_0_02"},
		{Name: "inner_lr_w", Shape: []int{D, spec.Heads}, InitMode: "ttt_normal_0_02"},
		{Name: "inner_lr_bias", Shape: []int{spec.Heads}, InitZero: true},
		{Name: "inner_token_coeff", Shape: []int{chunk}, InitZero: true},
		{Name: "inner_w1", Shape: []int{spec.Heads * headDim, hidden}, InitMode: "ttt_normal_0_02"},
		{Name: "inner_b1", Shape: []int{spec.Heads, hidden}, InitZero: true},
		{Name: "inner_w2", Shape: []int{spec.Heads * hidden, headDim}, InitMode: "ttt_normal_0_02"},
		{Name: "inner_b2", Shape: []int{spec.Heads, headDim}, InitZero: true},
		{Name: "inner_norm_scale", Shape: []int{spec.Heads, headDim}, IsNormScale: true, InitOne: true},
		{Name: "inner_norm_bias", Shape: []int{spec.Heads, headDim}, InitZero: true},
		{Name: "post_norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "post_norm_bias", Shape: []int{D}, InitZero: true},
		{Name: "w_out_gate", Shape: []int{D, D}, InitMode: "ttt_normal_0_02"},
		{Name: "w_out", Shape: []int{D, D}, InitMode: "ttt_normal_0_02"},
	}
	if blockScales {
		metas = append(metas, WeightMeta{Name: "ttt_mlp_scale", Shape: []int{D}, InitOne: true})
	}
	return metas, nil
}

func emitTTTMLPIR(prog *Program, spec BlockSpec, x string, wi, D, T, B, idx int, opts EmitOptions) (int, error) {
	if spec.Heads <= 0 || D%spec.Heads != 0 {
		return wi, fmt.Errorf("ttt_mlp requires model_dim divisible by heads")
	}
	headDim := D / spec.Heads
	if headDim%2 != 0 {
		return wi, fmt.Errorf("ttt_mlp requires even head_dim")
	}
	hidden, err := effectiveTTTMLPInnerHiddenDim(spec, D)
	if err != nil {
		return wi, err
	}
	prefix := tmpName(x+"_ttt_mlp", idx)
	xNorm := prefix + "_x_norm"
	qkRaw := prefix + "_qk_raw"
	qConv := prefix + "_q_conv"
	qBiased := prefix + "_q_biased"
	kConv := prefix + "_k_conv"
	kBiased := prefix + "_k_biased"
	vRaw := prefix + "_v_raw"
	lrRaw := prefix + "_lr_raw"
	lrLogits := prefix + "_lr_logits"
	scanOut := prefix + "_scan_out"
	innerBefore := prefix + "_inner_loss_before"
	innerAfter := prefix + "_inner_loss_after"
	updateNorm := prefix + "_update_norm"
	stateDrift := prefix + "_state_drift"
	lrMean := prefix + "_lr_mean"
	lrMin := prefix + "_lr_min"
	lrMax := prefix + "_lr_max"
	lrScale := prefix + "_lr_scale"
	merged := prefix + "_merged"
	postNorm := prefix + "_post_norm"
	gateRaw := prefix + "_gate_raw"
	gate := prefix + "_gate"
	gated := prefix + "_gated"
	delta := prefix + "_delta"

	prog.RMSNorm(x, weightName(wi), xNorm, 1e-6)
	wi++
	prog.MatMul(xNorm, weightName(wi), qkRaw)
	wi++
	prog.DepthwiseConv1DReversed(qkRaw, weightName(wi), qConv, B, T, D, defaultTTTMLPConvKernel)
	wi++
	prog.Add(qConv, weightName(wi), qBiased)
	wi++
	prog.DepthwiseConv1DReversed(qkRaw, weightName(wi), kConv, B, T, D, defaultTTTMLPConvKernel)
	wi++
	prog.Add(kConv, weightName(wi), kBiased)
	wi++
	prog.MatMul(xNorm, weightName(wi), vRaw)
	wi++
	prog.MatMul(xNorm, weightName(wi), lrRaw)
	wi++
	prog.Add(lrRaw, weightName(wi), lrLogits)
	wi++

	prog.Slice("ttt_inner_lr_scale", opts.BlockIndex, opts.BlockIndex+1, 1, 0, lrScale)
	prog.TTTMLPScan(
		qBiased, kBiased, vRaw, lrLogits, lrScale,
		weightName(wi), weightName(wi+1), weightName(wi+2), weightName(wi+3),
		weightName(wi+4), weightName(wi+5), weightName(wi+6),
		scanOut, innerBefore, innerAfter, updateNorm, stateDrift, lrMean, lrMin, lrMax,
		B, T, spec.Heads, headDim, hidden, effectiveTTTMLPChunkSize(spec),
		float32(effectiveTTTMLPInnerLRBase(spec)))
	wi += 7

	prog.Reshape(scanOut, []int{B * T, D}, merged)
	prog.LayerNorm(merged, weightName(wi), weightName(wi+1), postNorm, 1e-6)
	wi += 2
	prog.MatMul(xNorm, weightName(wi), gateRaw)
	wi++
	prog.GELU(gateRaw, gate)
	prog.Mul(postNorm, gate, gated)
	prog.MatMul(gated, weightName(wi), delta)
	wi++
	if opts.BlockScales {
		scaled := prefix + "_scaled"
		prog.Mul(delta, weightName(wi), scaled)
		wi++
		delta = scaled
	}
	prog.Add(x, delta, x)

	for _, output := range []struct {
		name string
		src  string
	}{
		{"ttt_inner_loss_before", innerBefore},
		{"ttt_inner_loss_after", innerAfter},
		{"ttt_inner_update_norm", updateNorm},
		{"ttt_state_drift", stateDrift},
		{"ttt_inner_lr_mean", lrMean},
		{"ttt_inner_lr_min", lrMin},
		{"ttt_inner_lr_max", lrMax},
	} {
		// Multiple TTT blocks receive stable block-qualified diagnostics.
		name := fmt.Sprintf("block_%d_%s", opts.BlockIndex, output.name)
		prog.ScalarMul(output.src, 1, name)
		prog.DeclareOutput(name, TensorFloat32, []int{1})
	}
	return wi, nil
}

func estimateTTTMLPBlockFLOPs(block BlockSpec, B, T, D int) int64 {
	if block.Heads <= 0 || D%block.Heads != 0 {
		return 0
	}
	headDim := D / block.Heads
	hidden, err := effectiveTTTMLPInnerHiddenDim(block, D)
	if err != nil {
		return 0
	}
	outerProj := 2 * i64(B) * i64(T) * i64(D) * i64(D) * 4
	lrProj := 2 * i64(B) * i64(T) * i64(D) * i64(block.Heads)
	conv := 2 * i64(B) * i64(T) * i64(D) * defaultTTTMLPConvKernel
	inner := 8 * i64(B) * i64(T) * i64(block.Heads) * i64(headDim) * i64(hidden)
	chunkAttn := 4 * i64(B) * i64(T) * i64(block.Heads) * i64(effectiveTTTMLPChunkSize(block)) * i64(headDim+hidden)
	return outerProj + lrProj + conv + inner + chunkAttn
}

func tttMLPInnerLRScale(spec BlockSpec, step int) float32 {
	base := effectiveTTTMLPInnerLRBase(spec)
	init := effectiveTTTMLPInnerLRInit(spec)
	warmup := effectiveTTTMLPInnerLRWarmupSteps(spec)
	if warmup <= 1 || step >= warmup-1 {
		return 1
	}
	progress := float64(max(step, 0)) / float64(warmup-1)
	return float32(init/base + (1-init/base)*math.Max(0, math.Min(1, progress)))
}

func containsTTTMLP(blocks []BlockSpec) bool {
	for _, block := range blocks {
		if blockTypeKey(block) == "ttt_mlp" {
			return true
		}
	}
	return false
}

// TTTMLPInnerLRScalesForStep returns one stable-shape scale per configured
// block. Non-TTT entries are one so the runtime input layout is independent of
// which blocks consume it.
func TTTMLPInnerLRScalesForStep(blocks []BlockSpec, step int) []float32 {
	if !containsTTTMLP(blocks) {
		return nil
	}
	out := make([]float32, len(blocks))
	for i, block := range blocks {
		out[i] = 1
		if blockTypeKey(block) == "ttt_mlp" {
			out[i] = tttMLPInnerLRScale(block, step)
		}
	}
	return out
}

// TTTMLPRecurrentStateCountFromConfig returns the number of float scalars
// copied per sequence row for all TTT-MLP inner learners.
func TTTMLPRecurrentStateCountFromConfig(cfg *ArchConfig) (int64, bool, error) {
	if cfg == nil {
		return 0, false, fmt.Errorf("nil config")
	}
	var total int64
	found := false
	for _, block := range cfg.Blocks {
		if blockTypeKey(block) != "ttt_mlp" {
			continue
		}
		found = true
		hidden, err := effectiveTTTMLPInnerHiddenDim(block, cfg.ModelDim)
		if err != nil {
			return 0, false, err
		}
		headDim := cfg.ModelDim / block.Heads
		// W1 + b1 + W2 + b2 for every head.
		total += int64(block.Heads) * int64(2*headDim*hidden+hidden+headDim)
	}
	return total, found, nil
}
