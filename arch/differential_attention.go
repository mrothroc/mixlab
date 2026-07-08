package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	diffLambdaInitMode    = "diff_lambda_normal_0_1"
	diffLambdaExpClampAbs = 5.0
	diffAttentionSubLNEps = 1e-3
)

func differentialAttentionEnabled(block BlockSpec) bool {
	return blockTypeKey(block) == "plain" && block.DifferentialAttention
}

func differentialAttentionHeadWidth(modelDim int, block BlockSpec) (int, error) {
	heads := block.Heads
	if heads <= 0 {
		return 0, fmt.Errorf("differential_attention requires heads > 0")
	}
	if modelDim <= 0 || modelDim%heads != 0 {
		return 0, fmt.Errorf("differential_attention requires model_dim=%d divisible by heads=%d; for a baseline with N heads, set differential heads to N/2", modelDim, heads)
	}
	headWidth := modelDim / heads
	if headWidth%2 != 0 {
		return 0, fmt.Errorf("differential_attention requires model_dim/heads=%d to be even; for a baseline with N heads, set differential heads to N/2", headWidth)
	}
	return headWidth, nil
}

func differentialAttentionSubDim(modelDim int, block BlockSpec) (int, error) {
	headWidth, err := differentialAttentionHeadWidth(modelDim, block)
	if err != nil {
		return 0, err
	}
	return headWidth / 2, nil
}

func differentialLambdaInit(block BlockSpec, blockIndex int) float64 {
	return effectiveDifferentialLambdaInit(block.DifferentialLambdaInit, blockIndex)
}

// EffectiveDifferentialLambdaInit returns the DIFF Transformer lambda_init for
// a plain block at zero-based blockIndex. The schedule itself uses one-based
// layer depth, so blockIndex 0 maps to depth 1.
func EffectiveDifferentialLambdaInit(block BlockSpec, blockIndex int) float64 {
	return differentialLambdaInit(block, blockIndex)
}

func effectiveDifferentialLambdaInit(override *float64, blockIndex int) float64 {
	if override != nil {
		return *override
	}
	if blockIndex < 0 {
		blockIndex = 0
	}
	depth := blockIndex + 1
	return 0.8 - 0.6*math.Exp(-0.3*float64(depth))
}

func validateDifferentialAttention(cfg *ArchConfig, source string) error {
	if cfg == nil {
		return nil
	}
	for i, block := range cfg.Blocks {
		if !differentialAttentionEnabled(block) {
			continue
		}
		subDim, err := differentialAttentionSubDim(cfg.ModelDim, block)
		if err != nil {
			return fmt.Errorf("config %q blocks[%d] type=plain %w", source, i, err)
		}
		if block.DifferentialLambdaInit != nil && (math.IsNaN(*block.DifferentialLambdaInit) || math.IsInf(*block.DifferentialLambdaInit, 0)) {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid differential_lambda_init=%g (must be finite)", source, i, *block.DifferentialLambdaInit)
		}
		if block.RopeDims < 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid rope_dims=%d (must be > 0 when set)", source, i, block.RopeDims)
		}
		if block.RopeDims%2 != 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid rope_dims=%d (must be even)", source, i, block.RopeDims)
		}
		if block.RopeDims > subDim {
			return fmt.Errorf("config %q blocks[%d] type=plain has invalid rope_dims=%d (must be <= differential sub-head dim=%d)", source, i, block.RopeDims, subDim)
		}
		if normalizeRelativeAttention(block.RelativeAttention) != "" && normalizeRelativeAttention(block.RelativeAttention) != RelativeAttentionNone {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with relative_attention", source, i)
		}
		if block.QKNorm {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with qk_norm in v1", source, i)
		}
		if block.QKGain > 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with qk_gain in v1", source, i)
		}
		if block.AttnValueGate {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with attn_value_gate in v1", source, i)
		}
		if block.SparseAttnGate {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with sparse_attn_gate in v1", source, i)
		}
		if block.XSA {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with xsa in v1", source, i)
		}
		if block.KVSource > 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with kv_source in v1", source, i)
		}
		if block.KVHeads != 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with kv_heads in v1", source, i)
		}
		if block.ParallelGroup > 0 {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with parallel_group in v1", source, i)
		}
		if mode := normalizePlainAttnPostNorm(block.AttnPostNorm); mode != PlainAttnPostNormInherit && mode != PlainAttnPostNormNone {
			return fmt.Errorf("config %q blocks[%d] type=plain cannot combine differential_attention with attn_post_norm=%q in v1", source, i, block.AttnPostNorm)
		}
		if mode := strings.ToLower(strings.TrimSpace(cfg.EffectiveNormPlacement())); mode == NormPlacementPost || mode == NormPlacementSandwich {
			return fmt.Errorf("config %q blocks[%d] type=plain differential_attention does not support inherited attn_post_norm from norm_placement=%q in v1", source, i, cfg.EffectiveNormPlacement())
		}
	}
	return nil
}

func emitDifferentialAttentionFlatIR(prog *Program, prefix, qh, kh, vh string, wi, B, T, H, headWidth int, attnDropout float32, attentionMask string, windowSize int, segmentMask bool, ropeDims int, ropeConvention, positionalEmbedding string, lambdaInit float32) (string, int, error) {
	if headWidth <= 0 || headWidth%2 != 0 {
		return "", wi, fmt.Errorf("differential_attention requires even head width, got %d", headWidth)
	}
	subDim := headWidth / 2
	scale := float32(1.0 / math.Sqrt(float64(subDim)))

	q1 := prefix + "_diff_q1"
	q2 := prefix + "_diff_q2"
	k1 := prefix + "_diff_k1"
	k2 := prefix + "_diff_k2"
	prog.Slice(qh, 0, subDim, 1, 3, q1)
	prog.Slice(qh, subDim, headWidth, 1, 3, q2)
	prog.Slice(kh, 0, subDim, 1, 3, k1)
	prog.Slice(kh, subDim, headWidth, 1, 3, k2)

	q1Score := q1
	k1Score := k1
	q2Score := q2
	k2Score := k2
	if normalizePositionalEmbedding(positionalEmbedding) == PositionalEmbeddingRope {
		q1Rot := prefix + "_diff_q1_rot"
		k1Rot := prefix + "_diff_k1_rot"
		q2Rot := prefix + "_diff_q2_rot"
		k2Rot := prefix + "_diff_k2_rot"
		emitRoPEForConvention(prog, q1, k1, q1Rot, k1Rot, T, subDim, ropeDims, ropeConvention)
		emitRoPEForConvention(prog, q2, k2, q2Rot, k2Rot, T, subDim, ropeDims, ropeConvention)
		q1Score = q1Rot
		k1Score = k1Rot
		q2Score = q2Rot
		k2Score = k2Rot
	}

	attn1, err := emitDifferentialAttentionMapIR(prog, prefix+"_diff_1", q1Score, k1Score, B, T, scale, attentionMask, windowSize, segmentMask)
	if err != nil {
		return "", wi, err
	}
	attn2, err := emitDifferentialAttentionMapIR(prog, prefix+"_diff_2", q2Score, k2Score, B, T, scale, attentionMask, windowSize, segmentMask)
	if err != nil {
		return "", wi, err
	}

	lambda, nextWI := emitDifferentialLambdaIR(prog, prefix, wi, subDim, lambdaInit)
	wi = nextWI
	attn2Weighted := prefix + "_diff_attn2_weighted"
	diffAttn := prefix + "_diff_attn"
	diffAttnDrop := prefix + "_diff_attn_dropout"
	prog.Mul(attn2, lambda, attn2Weighted)
	prog.Sub(attn1, attn2Weighted, diffAttn)
	if attnDropout > 0 {
		prog.Dropout(diffAttn, attnDropout, diffAttnDrop)
		diffAttn = diffAttnDrop
	}

	ctx := prefix + "_diff_ctx"
	ctxFlatForNorm := prefix + "_diff_ctx_flat_for_subln"
	ctxNormFlat := prefix + "_diff_ctx_subln_flat"
	ctxNorm := prefix + "_diff_ctx_subln"
	ctxScaled := prefix + "_diff_ctx_scaled"
	ctxT := prefix + "_diff_ctx_t"
	flat := prefix + "_diff_flat"
	prog.MatMul(diffAttn, vh, ctx)
	prog.Reshape(ctx, []int{B * H * T, headWidth}, ctxFlatForNorm)
	prog.RMSNorm(ctxFlatForNorm, weightName(wi), ctxNormFlat, diffAttentionSubLNEps)
	wi++
	prog.Reshape(ctxNormFlat, []int{B, H, T, headWidth}, ctxNorm)
	prog.ScalarMul(ctxNorm, 1-lambdaInit, ctxScaled)
	prog.Transpose(ctxScaled, []int{0, 2, 1, 3}, ctxT)
	prog.Reshape(ctxT, []int{B * T, H * headWidth}, flat)
	return flat, wi, nil
}

func emitDifferentialAttentionMapIR(prog *Program, prefix, q, k string, B, T int, scale float32, attentionMask string, windowSize int, segmentMask bool) (string, error) {
	kt := prefix + "_kt"
	scores := prefix + "_scores"
	scaled := prefix + "_scaled"
	masked := prefix + "_masked"
	attn := prefix + "_attn"
	prog.Transpose(k, []int{0, 1, 3, 2}, kt)
	prog.MatMul(q, kt, scores)
	prog.ScalarMul(scores, scale, scaled)
	maskedScores, err := emitPlainAttentionMaskIR(prog, scaled, masked, attentionMask, B, T, windowSize, segmentMask)
	if err != nil {
		return "", err
	}
	prog.Softmax(maskedScores, -1, attn)
	return attn, nil
}

func emitDifferentialLambdaIR(prog *Program, prefix string, wi, subDim int, lambdaInit float32) (string, int) {
	lambdaQ1 := weightName(wi)
	lambdaK1 := weightName(wi + 1)
	lambdaQ2 := weightName(wi + 2)
	lambdaK2 := weightName(wi + 3)
	wi += 4

	dot1 := emitDifferentialLambdaDotIR(prog, prefix+"_lambda_1", lambdaQ1, lambdaK1, subDim)
	dot2 := emitDifferentialLambdaDotIR(prog, prefix+"_lambda_2", lambdaQ2, lambdaK2, subDim)
	dot1Clamped := prefix + "_lambda_dot1_clamped"
	dot2Clamped := prefix + "_lambda_dot2_clamped"
	exp1 := prefix + "_lambda_exp1"
	exp2 := prefix + "_lambda_exp2"
	diff := prefix + "_lambda_diff"
	initTensor := prefix + "_lambda_init"
	lambda := prefix + "_lambda"
	lambda4 := prefix + "_lambda4"
	// Keep the exp reparameterization finite after early optimizer shocks while
	// leaving normal small-dot initial behavior unchanged.
	prog.Clamp(dot1, -diffLambdaExpClampAbs, diffLambdaExpClampAbs, dot1Clamped)
	prog.Clamp(dot2, -diffLambdaExpClampAbs, diffLambdaExpClampAbs, dot2Clamped)
	prog.Exp(dot1Clamped, exp1)
	prog.Exp(dot2Clamped, exp2)
	prog.Sub(exp1, exp2, diff)
	prog.Full([]int{1, 1}, lambdaInit, initTensor)
	prog.Add(diff, initTensor, lambda)
	prog.Reshape(lambda, []int{1, 1, 1, 1}, lambda4)
	return lambda4, wi
}

func emitDifferentialLambdaDotIR(prog *Program, prefix, a, b string, subDim int) string {
	aRow := prefix + "_a_row"
	bCol := prefix + "_b_col"
	dot := prefix + "_dot"
	prog.Reshape(a, []int{1, subDim}, aRow)
	prog.Reshape(b, []int{subDim, 1}, bCol)
	prog.MatMul(aRow, bCol, dot)
	return dot
}
