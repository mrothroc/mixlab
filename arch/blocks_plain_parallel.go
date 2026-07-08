package arch

import (
	"fmt"
	"math"
)

func emitPlainAttentionParallelDeltaIRWithDropoutEx(prog *Program, x, xNorm string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, dropout, attnDropout float32, qkGain float64, qkNorm bool, differentialAttention bool, differentialLambdaInit *float64, ropeDims int, ropeConvention string, attnBias, attnValueGate bool, xsa, sparseAttnGate bool, windowSize int, attentionMask, relativeAttention string, relativeWindow int, relativeParameterization string, ffnActivation string, norm NormSpec, ffnPreNorm, ffnBias bool, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, segmentMask bool) (string, int, error) {
	_ = mlpMult
	norm = normSpecOrDefault(norm)
	ffnActivation = normalizePlainFFNActivation(ffnActivation)
	switch ffnActivation {
	case PlainFFNActivationSiLU, PlainFFNActivationGEGLU, PlainFFNActivationSwiGLU, PlainFFNActivationGELU, PlainFFNActivationGELUNew:
	default:
		return "", wi, fmt.Errorf("invalid plain ffn_activation=%q", ffnActivation)
	}
	if H <= 0 || D <= 0 || D%H != 0 {
		return "", wi, fmt.Errorf("invalid attention dimensions D=%d H=%d", D, H)
	}
	if differentialAttention {
		if headWidth := D / H; headWidth%2 != 0 {
			return "", wi, fmt.Errorf("differential_attention requires even head width, got %d", headWidth)
		}
		if kvH > 0 && kvH != H {
			return "", wi, fmt.Errorf("differential_attention does not support kv_heads in v1")
		}
		if qkNorm || qkGain > 0 || attnValueGate || xsa || sparseAttnGate {
			return "", wi, fmt.Errorf("differential_attention cannot be combined with qk_norm, qk_gain, attn_value_gate, xsa, or sparse_attn_gate in v1")
		}
		if normalizeRelativeAttention(relativeAttention) != "" && normalizeRelativeAttention(relativeAttention) != RelativeAttentionNone {
			return "", wi, fmt.Errorf("differential_attention cannot be combined with relative_attention in v1")
		}
	}
	if qkGain < 0 {
		return "", wi, fmt.Errorf("invalid qk_gain=%g", qkGain)
	}
	if windowSize < 0 {
		return "", wi, fmt.Errorf("invalid window_size=%d", windowSize)
	}
	attentionMask = normalizeAttentionMask(attentionMask)
	if attentionMask == "" {
		attentionMask = AttentionMaskCausal
	}
	if windowSize > 0 && attentionMask != AttentionMaskCausal {
		return "", wi, fmt.Errorf("window_size requires causal attention mask")
	}
	kvH, err := normalizePlainKVHeads(H, kvH)
	if err != nil {
		return "", wi, err
	}
	headDim := D / H
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvProjDim := kvH * headDim
	groupSize := H / kvH

	prefix := tmpName(x+"_parallel_attn", idx)
	q := prefix + "_q"
	k := prefix + "_k"
	v := prefix + "_v"
	vRaw := prefix + "_v_raw"
	valueGate := prefix + "_value_gate"
	valueGateAct := valueGate + "_act"
	q4 := q + "4"
	k4 := k + "4"
	v4 := v + "4"
	qh := q + "h"
	kh := k + "h"
	vh := v + "h"
	scores := prefix + "_scores"
	scaled := ""
	masked := scores + "_masked"
	attn := prefix + "_attn"
	attnDrop := prefix + "_attn_dropout"
	ctx := prefix + "_ctx"
	ctxXSA := prefix + "_ctx_xsa"
	gateIn := prefix + "_gate_in"
	gateWT := prefix + "_gate_wt"
	gateLogits := prefix + "_gate_logits"
	gateAct := prefix + "_gate_act"
	gate4BTH := prefix + "_gate4_bth"
	gate4 := prefix + "_gate4"
	ctxGated := prefix + "_ctx_gated"
	ctxT := prefix + "_ctx_t"
	flat := prefix + "_flat"
	flatGated := prefix + "_flat_value_gated"
	proj := prefix + "_proj"
	projScaled := prefix + "_proj_scaled"
	projDrop := prefix + "_proj_dropout"

	qWeightName, qBiasName, wi := emitLinearProjectionIR(prog, xNorm, wi, attnBias, q)
	kWeightName, kBiasName, wi := emitLinearProjectionIR(prog, xNorm, wi, attnBias, k)
	if attnValueGate {
		_, _, wi = emitLinearProjectionIR(prog, xNorm, wi, attnBias, vRaw)
		prog.Slice(vRaw, 0, kvProjDim, 1, 1, v)
		prog.Slice(vRaw, kvProjDim, kvProjDim+D, 1, 1, valueGate)
		prog.GELU(valueGate, valueGateAct)
	} else {
		_, _, wi = emitLinearProjectionIR(prog, xNorm, wi, attnBias, v)
	}

	prog.Reshape(q, []int{B, T, H, headDim}, q4)
	prog.Transpose(q4, []int{0, 2, 1, 3}, qh)
	prog.Reshape(k, []int{B, T, kvH, headDim}, k4)
	prog.Transpose(k4, []int{0, 2, 1, 3}, kh)
	prog.Reshape(v, []int{B, T, kvH, headDim}, v4)
	prog.Transpose(v4, []int{0, 2, 1, 3}, vh)
	if kvProjDim != D {
		khExp := kh + "_exp"
		khOnes := kh + "_ones"
		khRep := kh + "_rep"
		vhExp := vh + "_exp"
		vhOnes := vh + "_ones"
		vhRep := vh + "_rep"

		prog.Reshape(kh, []int{B, kvH, 1, T, headDim}, khExp)
		prog.Full([]int{1, 1, groupSize, 1, 1}, 1.0, khOnes)
		prog.Mul(khExp, khOnes, khRep)
		prog.Reshape(khRep, []int{B, H, T, headDim}, kh)

		prog.Reshape(vh, []int{B, kvH, 1, T, headDim}, vhExp)
		prog.Full([]int{1, 1, groupSize, 1, 1}, 1.0, vhOnes)
		prog.Mul(vhExp, vhOnes, vhRep)
		prog.Reshape(vhRep, []int{B, H, T, headDim}, vh)
	}

	if differentialAttention {
		lambdaInit := float32(effectiveDifferentialLambdaInit(differentialLambdaInit, idx))
		flat, wi, err = emitDifferentialAttentionFlatIR(prog, prefix, qh, kh, vh, wi, B, T, H, headDim, attnDropout, attentionMask, windowSize, segmentMask, ropeDims, ropeConvention, positionalEmbedding, lambdaInit)
		if err != nil {
			return "", wi, err
		}
	} else {
		qh, kh, wi = emitQKNormIR(prog, qh, kh, wi, qkNorm, true)
		scaled, _, wi, err = emitPlainProjectedAttentionScoresIR(prog, prefix, qh, kh, wi, B, H, kvH, D, T, headDim, scale, qkGain, ropeDims, ropeConvention, positionalEmbedding, relativeAttention, relativeWindow, relativeParameterization, qWeightName, qBiasName, kWeightName, kBiasName, sharedRel)
		if err != nil {
			return "", wi, err
		}
		maskedScores, err := emitPlainAttentionMaskIR(prog, scaled, masked, attentionMask, B, T, windowSize, segmentMask)
		if err != nil {
			return "", wi, err
		}
		prog.Softmax(maskedScores, -1, attn)
		if attnDropout > 0 {
			prog.Dropout(attn, attnDropout, attnDrop)
			attn = attnDrop
		}
		prog.MatMul(attn, vh, ctx)
		if xsa {
			prog.XSAProject(ctx, vh, ctxXSA)
			ctx = ctxXSA
		}
		if sparseAttnGate {
			gateDim := plainSparseAttnGateWidth(D)
			prog.Slice(x, 0, gateDim, 1, 1, gateIn)
			prog.Transpose(weightName(wi), []int{1, 0}, gateWT)
			wi++
			prog.MatMul(gateIn, gateWT, gateLogits)
			prog.Sigmoid(gateLogits, gateAct)
			prog.Reshape(gateAct, []int{B, T, H, 1}, gate4BTH)
			prog.Transpose(gate4BTH, []int{0, 2, 1, 3}, gate4)
			prog.Mul(ctx, gate4, ctxGated)
			ctx = ctxGated
		}
		prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
		prog.Reshape(ctxT, []int{B * T, D}, flat)
		if attnValueGate {
			prog.Mul(flat, valueGateAct, flatGated)
			flat = flatGated
		}
	}

	_, _, wi = emitLinearProjectionIR(prog, flat, wi, attnBias, proj)
	if blockScales {
		prog.Mul(proj, weightName(wi), projScaled)
		wi++
		proj = projScaled
	}
	if dropout > 0 {
		prog.Dropout(proj, dropout, projDrop)
		proj = projDrop
	}

	ffIn := prefix + "_ff_in"
	ffGate := prefix + "_ff_gate"
	ffGateAct := ffGate + "_act"
	ff1 := prefix + "_ff1"
	ffInNorm := prefix + "_ffn_x_norm"
	ffAct := prefix + "_ff_act"
	ff2 := prefix + "_ff2"
	ff2Scaled := prefix + "_ff2_scaled"
	ff2Drop := prefix + "_ff2_dropout"
	state := prefix + "_state"
	prog.Add(x, proj, ffIn)
	if ffnPreNorm {
		var err error
		wi, err = emitNamedNormIR(prog, ffIn, wi, ffInNorm, norm)
		if err != nil {
			return "", wi, err
		}
		ffIn = ffInNorm
	}
	switch ffnActivation {
	case PlainFFNActivationSiLU, PlainFFNActivationGELU, PlainFFNActivationGELUNew:
		_, _, wi = emitLinearProjectionIR(prog, ffIn, wi, ffnBias, ff1)
		switch ffnActivation {
		case PlainFFNActivationSiLU:
			prog.SiLU(ff1, ffAct)
		case PlainFFNActivationGELU:
			prog.GELUExact(ff1, ffAct)
		case PlainFFNActivationGELUNew:
			prog.GELU(ff1, ffAct)
		}
	case PlainFFNActivationGEGLU, PlainFFNActivationSwiGLU:
		prog.MatMul(ffIn, weightName(wi), ffGate)
		wi++
		if ffnActivation == PlainFFNActivationGEGLU {
			prog.GELU(ffGate, ffGateAct)
		} else {
			prog.SiLU(ffGate, ffGateAct)
		}
		_, _, wi = emitLinearProjectionIR(prog, ffIn, wi, ffnBias, ff1)
		prog.Mul(ffGateAct, ff1, ffAct)
	default:
		return "", wi, fmt.Errorf("invalid plain ffn_activation=%q", ffnActivation)
	}
	_, _, wi = emitLinearProjectionIR(prog, ffAct, wi, ffnBias, ff2)
	if blockScales {
		prog.Mul(ff2, weightName(wi), ff2Scaled)
		wi++
		ff2 = ff2Scaled
	}
	if dropout > 0 {
		prog.Dropout(ff2, dropout, ff2Drop)
		ff2 = ff2Drop
	}
	prog.Add(ffIn, ff2, state)
	return state, wi, nil
}
