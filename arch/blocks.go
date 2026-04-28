package arch

import (
	"fmt"
	"math"
	"strings"
)

// emitPlainAttentionIR emits a plain causal self-attention block with
// RoPE, scaled dot-product attention, causal mask, and a feed-forward tail.
// When kvH < H, K/V use grouped-query attention and are repeated across query
// head groups before the attention matmul.
//
// Weight layout (7 weights per block, or 8 when qk_gain is enabled):
//
//	w[wi+0] = RMSNorm scale
//	w[wi+1] = Q projection
//	w[wi+2] = K projection
//	w[wi+3] = V projection
//	w[wi+4] = optional per-head QK gain
//	w[wi+4/5] = output projection
//	w[wi+5/6] = FF layer 1
//	w[wi+6/7] = FF layer 2
func emitPlainAttentionIR(prog *Program, x string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool) (int, error) { //nolint:unparam // x and B are fixed at IR build time by design
	return emitPlainAttentionIRWithWindow(prog, x, wi, H, kvH, D, T, B, idx, mlpMult, blockScales, 0)
}

func emitPlainAttentionIRWithWindow(prog *Program, x string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, windowSize int) (int, error) {
	return emitPlainAttentionIRWithDropout(prog, x, wi, H, kvH, D, T, B, idx, mlpMult, blockScales, 0, windowSize)
}

func emitPlainAttentionIRWithDropout(prog *Program, x string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, windowSize int) (int, error) {
	return emitPlainAttentionIRWithOptions(prog, x, wi, H, kvH, D, T, B, idx, mlpMult, blockScales, dropout, false, 0, 0, false, false, windowSize)
}

func emitPlainAttentionIRWithOptions(prog *Program, x string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, skipAttention bool, qkGain float64, ropeDims int, xsa, sparseAttnGate bool, windowSize int) (int, error) {
	return emitPlainAttentionIRWithKVOptions(prog, x, wi, H, kvH, D, T, B, idx, mlpMult, blockScales, dropout, skipAttention, qkGain, ropeDims, xsa, sparseAttnGate, windowSize, 0, nil, -1)
}

func emitPlainAttentionIRWithKVOptions(prog *Program, x string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, skipAttention bool, qkGain float64, ropeDims int, xsa, sparseAttnGate bool, windowSize int, kvSource int, kvCache map[int]BlockKVOutputs, blockIndex int) (int, error) {
	_ = mlpMult
	if H <= 0 || D <= 0 || D%H != 0 {
		return wi, fmt.Errorf("invalid attention dimensions D=%d H=%d", D, H)
	}
	if qkGain < 0 {
		return wi, fmt.Errorf("invalid qk_gain=%g", qkGain)
	}
	if windowSize < 0 {
		return wi, fmt.Errorf("invalid window_size=%d", windowSize)
	}
	kvH, err := normalizePlainKVHeads(H, kvH)
	if err != nil {
		return wi, err
	}
	headDim := D / H
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvProjDim := kvH * headDim
	groupSize := H / kvH
	reuseKV := kvSource > 0

	prefix := tmpName(x+"_attn", idx)
	xNorm := prefix + "_x_norm"
	q := prefix + "_q"
	k := prefix + "_k"
	v := prefix + "_v"
	q4 := q + "4"
	k4 := k + "4"
	v4 := v + "4"
	qh := q + "h"
	kh := k + "h"
	vh := v + "h"
	kt := k + "t"
	qhRot := q + "_rot"
	khRot := k + "_rot"
	scores := prefix + "_scores"
	scaled := scores + "_scaled"
	gain := scores + "_qk_gain"
	gained := scores + "_gained"
	masked := scores + "_masked"
	attn := prefix + "_attn"
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
	proj := prefix + "_proj"
	projScaled := prefix + "_proj_scaled"
	projDrop := prefix + "_proj_dropout"

	if skipAttention {
		wi += 3 // norm, q, output projection
		if !reuseKV {
			wi += 2 // k, v
		}
		if qkGain > 0 {
			wi++ // per-head QK gain
		}
		if sparseAttnGate {
			wi++ // narrow per-head attention output gate
		}
		if blockScales {
			wi++ // attention residual scale
		}
	} else {
		// Pre-attention RMSNorm
		prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
		wi++

		// Q projection
		prog.MatMul(xNorm, weightName(wi), q)
		wi++
		prog.Reshape(q, []int{B, T, H, headDim}, q4)
		prog.Transpose(q4, []int{0, 2, 1, 3}, qh)

		if reuseKV {
			if kvCache == nil {
				return wi, fmt.Errorf("kv_source=%d requires KV cache", kvSource)
			}
			src, ok := kvCache[kvSource-1]
			if !ok || src.K == "" || src.V == "" {
				return wi, fmt.Errorf("kv_source=%d references block without emitted KV tensors", kvSource)
			}
			khRot = src.K
			vh = src.V
		} else {
			// K/V projections
			prog.MatMul(xNorm, weightName(wi), k)
			wi++
			prog.MatMul(xNorm, weightName(wi), v)
			wi++

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

			// Rotary position embeddings
			prog.RoPE(qh, kh, qhRot, khRot, T, headDim, ropeDims, 10000.0)
			if kvCache != nil && blockIndex >= 0 {
				kvCache[blockIndex] = BlockKVOutputs{K: khRot, V: vh}
			}
		}
		if reuseKV {
			prog.ScalarMul(qh, 1.0, qhRot)
		}

		// Attention scores: Q @ K^T / sqrt(headDim)
		prog.Transpose(khRot, []int{0, 1, 3, 2}, kt)
		prog.MatMul(qhRot, kt, scores)
		prog.ScalarMul(scores, scale, scaled)
		if qkGain > 0 {
			prog.Reshape(weightName(wi), []int{1, H, 1, 1}, gain)
			wi++
			prog.Mul(scaled, gain, gained)
			scaled = gained
		}

		// Causal mask + softmax
		prog.CausalMask(scaled, T, windowSize, masked)
		prog.Softmax(masked, -1, attn)

		// Attention output: attn @ V, then transpose back
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

		// Output projection + residual
		prog.MatMul(flat, weightName(wi), proj)
		wi++
		if blockScales {
			prog.Mul(proj, weightName(wi), projScaled)
			wi++
			proj = projScaled
		}
		if dropout > 0 {
			prog.Dropout(proj, dropout, projDrop)
			proj = projDrop
		}
		prog.Add(x, proj, x)
	}

	// Feed-forward tail: ff1 -> SiLU -> ff2 -> residual
	ff1 := prefix + "_ff1"
	ffAct := prefix + "_ff_act"
	ff2 := prefix + "_ff2"
	ff2Scaled := prefix + "_ff2_scaled"
	ff2Drop := prefix + "_ff2_dropout"
	prog.MatMul(x, weightName(wi), ff1)
	wi++
	prog.SiLU(ff1, ffAct)
	prog.MatMul(ffAct, weightName(wi), ff2)
	wi++
	if blockScales {
		prog.Mul(ff2, weightName(wi), ff2Scaled)
		wi++
		ff2 = ff2Scaled
	}
	if dropout > 0 {
		prog.Dropout(ff2, dropout, ff2Drop)
		ff2 = ff2Drop
	}
	prog.Add(x, ff2, x)

	return wi, nil
}

func plainSparseAttnGateWidth(dim int) int {
	if dim <= 0 {
		return 1
	}
	if dim < 12 {
		return dim
	}
	return 12
}

func emitTokenShiftIR(prog *Program, x, mu, output string, B, T, D int, prefix string) {
	muSig := prefix + "_mu_sig"
	one := prefix + "_one"
	muKeep := prefix + "_mu_keep"
	x3 := prefix + "_x3"
	prevTail := prefix + "_prev_tail"
	zeroRow := prefix + "_zero_row"
	prev3 := prefix + "_prev3"
	prev := prefix + "_prev"
	currMix := prefix + "_curr_mix"
	prevMix := prefix + "_prev_mix"

	prog.Sigmoid(mu, muSig)
	prog.Full([]int{D}, 1.0, one)
	prog.Sub(one, muSig, muKeep)

	prog.Reshape(x, []int{B, T, D}, x3)
	prog.Slice(x3, 0, T-1, 1, 1, prevTail)
	prog.Full([]int{B, 1, D}, 0.0, zeroRow)
	prog.Concat(zeroRow, prevTail, 1, prev3)
	prog.Reshape(prev3, []int{B * T, D}, prev)

	prog.Mul(x, muKeep, currMix)
	prog.Mul(prev, muSig, prevMix)
	prog.Add(currMix, prevMix, output)
}

func repeatRowsIR(prog *Program, x string, repeats int, output string) {
	if repeats <= 1 {
		prog.ScalarMul(x, 1.0, output)
		return
	}
	current := x
	for i := 1; i < repeats; i++ {
		next := output
		if i < repeats-1 {
			next = fmt.Sprintf("%s_rep_%d", output, i)
		}
		prog.Concat(current, x, 0, next)
		current = next
	}
}

// emitTokenBlendIR emits a learned token blending gate over adjacent positions.
//
// Weight layout (1 weight per block):
//
//	w[wi+0] = W_gate [D, D]
//
// Forward pass:
//
//	gate   = sigmoid(x @ W_gate)
//	x_prev = token_shift(x, 1)
//	x      = (1 - gate) * x + gate * x_prev
func emitTokenBlendIR(prog *Program, x string, wi, D, T, B, idx int) (int, error) {
	if D <= 0 {
		return wi, fmt.Errorf("token_blend requires D > 0, got %d", D)
	}
	if T <= 0 {
		return wi, fmt.Errorf("token_blend requires T > 0, got %d", T)
	}
	if B <= 0 {
		return wi, fmt.Errorf("token_blend requires B > 0, got %d", B)
	}

	prefix := tmpName(x+"_token_blend", idx)
	gateRaw := prefix + "_gate_raw"
	gate := prefix + "_gate"
	one := prefix + "_one"
	gateKeep := prefix + "_gate_keep"
	x3 := prefix + "_x3"
	prevTail := prefix + "_prev_tail"
	zeroRow := prefix + "_zero_row"
	prev3 := prefix + "_prev3"
	prev := prefix + "_prev"
	currMix := prefix + "_curr_mix"
	prevMix := prefix + "_prev_mix"

	prog.MatMul(x, weightName(wi), gateRaw)
	prog.Sigmoid(gateRaw, gate)
	wi++

	prog.Reshape(x, []int{B, T, D}, x3)
	prog.Slice(x3, 0, T-1, 1, 1, prevTail)
	prog.Full([]int{B, 1, D}, 0.0, zeroRow)
	prog.Concat(zeroRow, prevTail, 1, prev3)
	prog.Reshape(prev3, []int{B * T, D}, prev)

	prog.Full([]int{D}, 1.0, one)
	prog.Sub(one, gate, gateKeep)
	prog.Mul(gateKeep, x, currMix)
	prog.Mul(gate, prev, prevMix)
	prog.Add(currMix, prevMix, x)

	return wi, nil
}

// emitSwiGLUIR emits a SwiGLU feed-forward block.
//
// Weight layout (4 weights per block):
//
//	w[wi+0] = RMSNorm scale
//	w[wi+1] = gate projection
//	w[wi+2] = up projection
//	w[wi+3] = down projection
func emitSwiGLUIR(prog *Program, x string, wi, idx int, mlpMult float64, blockScales bool) (int, error) {
	return emitSwiGLUIRWithDropout(prog, x, wi, idx, mlpMult, blockScales, 0)
}

func emitSwiGLUIRWithDropout(prog *Program, x string, wi, idx int, mlpMult float64, blockScales bool, dropout float32) (int, error) {
	_ = mlpMult
	prefix := tmpName(x+"_swiglu", idx)
	xNorm := prefix + "_x_norm"
	gate := prefix + "_gate"
	gateAct := gate + "_act"
	up := prefix + "_up"
	ff := prefix + "_ff"
	ffDown := prefix + "_down"
	ffScaled := prefix + "_scaled"
	ffDrop := prefix + "_dropout"

	// Pre-block RMSNorm
	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	// SwiGLU: sigmoid(gate_proj(x)) * up_proj(x), then down_proj
	prog.MatMul(xNorm, weightName(wi), gate)
	wi++
	prog.Sigmoid(gate, gateAct)
	prog.MatMul(xNorm, weightName(wi), up)
	wi++
	prog.Mul(gateAct, up, ff)
	prog.MatMul(ff, weightName(wi), ffDown)
	wi++
	if blockScales {
		prog.Mul(ffDown, weightName(wi), ffScaled)
		wi++
		ffDown = ffScaled
	}
	if dropout > 0 {
		prog.Dropout(ffDown, dropout, ffDrop)
		ffDown = ffDrop
	}
	prog.Add(x, ffDown, x)

	return wi, nil
}

// emitMLPIR emits a two-matrix feed-forward block:
// RMSNorm -> up projection -> activation -> down projection -> residual add.
//
// Weight layout (3 weights per block):
//
//	w[wi+0] = RMSNorm scale
//	w[wi+1] = up projection
//	w[wi+2] = down projection
func emitMLPIR(prog *Program, x string, wi, idx int, activation string, leakySlope float64) (int, error) {
	prefix := tmpName(x+"_mlp", idx)
	xNorm := prefix + "_x_norm"
	up := prefix + "_up"
	act := prefix + "_act"
	leaky := prefix + "_leaky"
	down := prefix + "_down"

	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++
	prog.MatMul(xNorm, weightName(wi), up)
	wi++

	switch strings.ToLower(strings.TrimSpace(activation)) {
	case "", "silu":
		prog.SiLU(up, act)
	case "gelu":
		prog.GELU(up, act)
	case "relu":
		prog.ReLU(up, act)
	case "leaky_relu_sq":
		slope := leakySlope
		if slope == 0 {
			slope = 0.5
		}
		prog.LeakyReLU(up, leaky, float32(slope))
		prog.Square(leaky, act)
	default:
		return wi, fmt.Errorf("unsupported mlp activation %q", activation)
	}

	prog.MatMul(act, weightName(wi), down)
	wi++
	prog.Add(x, down, x)

	return wi, nil
}

func emitPlainAttentionParallelDeltaIRWithDropout(prog *Program, x, xNorm string, wi, H, kvH, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, qkGain float64, ropeDims int, xsa, sparseAttnGate bool, windowSize int) (string, int, error) {
	_ = mlpMult
	if H <= 0 || D <= 0 || D%H != 0 {
		return "", wi, fmt.Errorf("invalid attention dimensions D=%d H=%d", D, H)
	}
	if qkGain < 0 {
		return "", wi, fmt.Errorf("invalid qk_gain=%g", qkGain)
	}
	if windowSize < 0 {
		return "", wi, fmt.Errorf("invalid window_size=%d", windowSize)
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
	q4 := q + "4"
	k4 := k + "4"
	v4 := v + "4"
	qh := q + "h"
	kh := k + "h"
	vh := v + "h"
	kt := k + "t"
	qhRot := q + "_rot"
	khRot := k + "_rot"
	scores := prefix + "_scores"
	scaled := scores + "_scaled"
	gain := scores + "_qk_gain"
	gained := scores + "_gained"
	masked := scores + "_masked"
	attn := prefix + "_attn"
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
	proj := prefix + "_proj"
	projScaled := prefix + "_proj_scaled"
	projDrop := prefix + "_proj_dropout"

	prog.MatMul(xNorm, weightName(wi), q)
	wi++
	prog.MatMul(xNorm, weightName(wi), k)
	wi++
	prog.MatMul(xNorm, weightName(wi), v)
	wi++

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

	prog.RoPE(qh, kh, qhRot, khRot, T, headDim, ropeDims, 10000.0)
	prog.Transpose(khRot, []int{0, 1, 3, 2}, kt)
	prog.MatMul(qhRot, kt, scores)
	prog.ScalarMul(scores, scale, scaled)
	if qkGain > 0 {
		prog.Reshape(weightName(wi), []int{1, H, 1, 1}, gain)
		wi++
		prog.Mul(scaled, gain, gained)
		scaled = gained
	}
	prog.CausalMask(scaled, T, windowSize, masked)
	prog.Softmax(masked, -1, attn)
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

	prog.MatMul(flat, weightName(wi), proj)
	wi++
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
	ff1 := prefix + "_ff1"
	ffAct := prefix + "_ff_act"
	ff2 := prefix + "_ff2"
	ff2Scaled := prefix + "_ff2_scaled"
	ff2Drop := prefix + "_ff2_dropout"
	state := prefix + "_state"
	prog.Add(x, proj, ffIn)
	prog.MatMul(ffIn, weightName(wi), ff1)
	wi++
	prog.SiLU(ff1, ffAct)
	prog.MatMul(ffAct, weightName(wi), ff2)
	wi++
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

func emitSwiGLUParallelDeltaIRWithDropout(prog *Program, xNorm string, wi, idx int, mlpMult float64, blockScales bool, dropout float32) (string, int) {
	_ = mlpMult
	prefix := tmpName(xNorm+"_parallel_swiglu", idx)
	gate := prefix + "_gate"
	gateAct := gate + "_act"
	up := prefix + "_up"
	ff := prefix + "_ff"
	ffDown := prefix + "_down"
	ffScaled := prefix + "_scaled"
	ffDrop := prefix + "_dropout"

	prog.MatMul(xNorm, weightName(wi), gate)
	wi++
	prog.Sigmoid(gate, gateAct)
	prog.MatMul(xNorm, weightName(wi), up)
	wi++
	prog.Mul(gateAct, up, ff)
	prog.MatMul(ff, weightName(wi), ffDown)
	wi++
	if blockScales {
		prog.Mul(ffDown, weightName(wi), ffScaled)
		wi++
		ffDown = ffScaled
	}
	if dropout > 0 {
		prog.Dropout(ffDown, dropout, ffDrop)
		ffDown = ffDrop
	}
	return ffDown, wi
}
