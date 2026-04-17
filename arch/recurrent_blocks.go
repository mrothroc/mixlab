package arch

import (
	"fmt"
	"math"
)

// emitCrossAttentionIR emits a cross-attention block where Q comes from the
// current stream and K/V come from a separate source stream. No causal mask
// is applied (the attending stream sees all positions in the source).
//
// Weight layout (7 weights per block):
//
//	w[wi+0] = RMSNorm scale
//	w[wi+1] = Q projection
//	w[wi+2] = K projection
//	w[wi+3] = V projection
//	w[wi+4] = output projection
//	w[wi+5] = FF layer 1
//	w[wi+6] = FF layer 2
func emitCrossAttentionIR(prog *Program, x, kvStream string, wi, H, D, Tq, Tkv, B, idx int) (int, error) {
	if H <= 0 || D <= 0 || D%H != 0 {
		return wi, fmt.Errorf("invalid cross-attention dimensions D=%d H=%d", D, H)
	}
	headDim := D / H
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	prefix := tmpName(x+"_xattn", idx)
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
	scores := prefix + "_scores"
	scaled := scores + "_scaled"
	attn := prefix + "_attn"
	ctx := prefix + "_ctx"
	ctxT := prefix + "_ctx_t"
	flat := prefix + "_flat"
	proj := prefix + "_proj"

	// Pre-attention RMSNorm (on query stream only)
	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	// Q projection from current stream, K/V from source stream
	prog.MatMul(xNorm, weightName(wi), q)
	wi++
	prog.MatMul(kvStream, weightName(wi), k)
	wi++
	prog.MatMul(kvStream, weightName(wi), v)
	wi++

	// Reshape Q: [B*Tq, D] -> [B, Tq, H, headDim] -> [B, H, Tq, headDim]
	prog.Reshape(q, []int{B, Tq, H, headDim}, q4)
	prog.Transpose(q4, []int{0, 2, 1, 3}, qh)

	// Reshape K/V: [B*Tkv, D] -> [B, Tkv, H, headDim] -> [B, H, Tkv, headDim]
	prog.Reshape(k, []int{B, Tkv, H, headDim}, k4)
	prog.Transpose(k4, []int{0, 2, 1, 3}, kh)
	prog.Reshape(v, []int{B, Tkv, H, headDim}, v4)
	prog.Transpose(v4, []int{0, 2, 1, 3}, vh)

	// Attention scores: Q @ K^T / sqrt(headDim) — no causal mask
	prog.Transpose(kh, []int{0, 1, 3, 2}, kt)
	prog.MatMul(qh, kt, scores)
	prog.ScalarMul(scores, scale, scaled)

	// Softmax (no masking — full cross-attention)
	prog.Softmax(scaled, -1, attn)

	// Attention output: attn @ V, then transpose back
	prog.MatMul(attn, vh, ctx)
	prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
	prog.Reshape(ctxT, []int{B * Tq, D}, flat)

	// Output projection + residual
	prog.MatMul(flat, weightName(wi), proj)
	wi++
	prog.Add(x, proj, x)

	// Feed-forward tail: ff1 -> SiLU -> ff2 -> residual
	ff1 := prefix + "_ff1"
	ffAct := prefix + "_ff_act"
	ff2 := prefix + "_ff2"
	prog.MatMul(x, weightName(wi), ff1)
	wi++
	prog.SiLU(ff1, ffAct)
	prog.MatMul(ffAct, weightName(wi), ff2)
	wi++
	prog.Add(x, ff2, x)

	return wi, nil
}

// emitMambaIR emits a Mamba selective-scan block.
//
// Weight layout (4 weights per block):
//
//	w[wi+0] = input projection   [D, 2*inner]
//	w[wi+1] = conv weight        [inner, inner]  (1D conv as matmul)
//	w[wi+2] = output projection  [inner, D]
//	w[wi+3] = scan decay         [inner]
//
// Forward pass:
//
//	proj = x @ in_proj                        [B*T, 2*inner]
//	z = proj[:, :inner]                       gating branch
//	h_in = proj[:, inner:2*inner]             recurrence branch
//	h_conv = h_in @ conv_w                    local mixing
//	h_scan = scan(h_conv, scan_decay)         gated recurrence (OpScan)
//	gate = silu(z)
//	h_gated = h_scan * gate
//	out = h_gated @ out_proj
//	x = x + out                               residual
func emitMambaIR(prog *Program, x string, wi, inner, T, B, idx int) (int, error) {
	if inner <= 0 {
		return wi, fmt.Errorf("mamba inner_dim must be > 0, got %d", inner)
	}

	prefix := tmpName(x+"_mamba", idx)
	proj := prefix + "_proj"
	z := prefix + "_z"
	hIn := prefix + "_h_in"
	hConv := prefix + "_h_conv"
	hScan := prefix + "_h_scan"
	gate := prefix + "_gate"
	hGated := prefix + "_h_gated"
	out := prefix + "_out"

	// Input projection: x [B*T, D] @ in_proj [D, 2*inner] -> proj [B*T, 2*inner]
	prog.MatMul(x, weightName(wi), proj)
	wi++

	// Split proj into z (gate) and h_in (recurrence) along last axis
	prog.Slice(proj, 0, inner, 1, 1, z)
	prog.Slice(proj, inner, inner*2, 1, 1, hIn)

	// Local mixing via conv weight: h_in [B*T, inner] @ conv_w [inner, inner]
	prog.MatMul(hIn, weightName(wi), hConv)
	wi++

	// Gated recurrence: scan(h_conv, decay) -> h_scan
	// Weight order: wi=out_proj, wi+1=scan_decay. Consume scan_decay first.
	prog.Scan(hConv, weightName(wi+1), hScan, B, T, inner)

	// Gate: silu(z)
	prog.SiLU(z, gate)

	// Gated output: h_scan * gate
	prog.Mul(hScan, gate, hGated)

	// Output projection: h_gated [B*T, inner] @ out_proj [inner, D]
	prog.MatMul(hGated, weightName(wi), out)
	wi += 2 // consume out_proj (wi) and scan_decay (wi+1)

	// Residual connection
	prog.Add(x, out, x)

	return wi, nil
}

// emitMamba3IR emits a Mamba-3 block with learned delta_t gating via sigmoid.
// This is the key difference from the plain mamba block: instead of a single
// gated recurrence, mamba3 modulates the scan input by sigmoid(dt_projection),
// giving the model learned temporal gating control.
//
// Weight layout (6 weights per block):
//
//	w[wi+0] = RMSNorm scale       [D]
//	w[wi+1] = gate projection     [D, D]   (Wz)
//	w[wi+2] = SSM input projection[D, D]   (Wx)
//	w[wi+3] = delta_t projection  [D, D]   (Wdt)
//	w[wi+4] = output projection   [D, D]   (Wo)
//	w[wi+5] = scan decay          [D]
//
// Forward pass:
//
//	x_norm  = RMSNorm(x)
//	z       = SiLU(x_norm @ Wz)              gating branch
//	h       = x_norm @ Wx                    SSM input
//	dt      = sigmoid(x_norm @ Wdt)          learned delta_t gate
//	h_gated = h * dt                         modulate by delta_t
//	h_scan  = scan(h_gated, decay)           gated recurrence (OpScan)
//	out     = (z * h_scan) @ Wo              gated output
//	x       = x + out                        residual
func emitMamba3IR(prog *Program, x string, wi, inner, T, B, idx int) (int, error) {
	if inner <= 0 {
		return wi, fmt.Errorf("mamba3 inner_dim must be > 0, got %d", inner)
	}

	prefix := tmpName(x+"_mamba3", idx)
	xNorm := prefix + "_x_norm"
	z := prefix + "_z"
	zAct := prefix + "_z_act"
	h := prefix + "_h"
	dtRaw := prefix + "_dt_raw"
	dt := prefix + "_dt"
	hGated := prefix + "_h_gated"
	hScan := prefix + "_h_scan"
	yGated := prefix + "_y_gated"
	out := prefix + "_out"

	// Pre-block RMSNorm
	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	// Gate branch: z = SiLU(x_norm @ Wz)
	prog.MatMul(xNorm, weightName(wi), z)
	wi++
	prog.SiLU(z, zAct)

	// SSM input: h = x_norm @ Wx
	prog.MatMul(xNorm, weightName(wi), h)
	wi++

	// Delta_t gating: dt = sigmoid(x_norm @ Wdt)
	prog.MatMul(xNorm, weightName(wi), dtRaw)
	wi++
	prog.Sigmoid(dtRaw, dt)

	// Modulate SSM input by delta_t
	prog.Mul(h, dt, hGated)

	// Gated recurrence: scan(h_gated, decay)
	// Weight order: wi=out_proj, wi+1=scan_decay. Consume scan_decay first.
	prog.Scan(hGated, weightName(wi+1), hScan, B, T, inner)

	// Gated output: z_act * h_scan
	prog.Mul(zAct, hScan, yGated)

	// Output projection
	prog.MatMul(yGated, weightName(wi), out)
	wi += 2 // consume out_proj (wi) and scan_decay (wi+1)

	// Residual connection
	prog.Add(x, out, x)

	return wi, nil
}

// emitRWKVIR emits a simplified RWKV-style linear-time block with
// channel-mixing and time-mixing using learned per-channel decay.
//
// Weight layout (10 weights per block):
//
//	w[wi+0] = Mu   [D]      per-channel shift ratio logits (time-mix)
//	w[wi+1] = Wr   [D, D]   receptance projection
//	w[wi+2] = Wk   [D, D]   key projection
//	w[wi+3] = Wv   [D, D]   value projection
//	w[wi+4] = W    [D]      per-channel decay logits
//	w[wi+5] = Wo   [D, D]   output projection
//	w[wi+6] = Mu2  [D]      per-channel shift ratio logits (channel-mix)
//	w[wi+7] = Wr2  [D, D]   channel-mix receptance projection
//	w[wi+8] = Wk2  [D, D]   channel-mix key projection
//	w[wi+9] = Wv2  [D, D]   channel-mix value projection
//
// Time mixing forward:
//
//	shifted  = TokenShift(x, mu)
//	r        = sigmoid(shifted @ Wr)
//	k        = shifted @ Wk
//	v        = shifted @ Wv
//	exp_k    = exp(k)
//	kv       = exp_k * v
//	decay    = exp(-exp(W))      (broadcast to [B*T, D])
//	num      = Scan(kv, decay)   running weighted sum
//	den      = Scan(exp_k, decay) running normalizer
//	wkv      = num / (den + 1e-6)
//	time_out = (r * wkv) @ Wo
//	x        = x + time_out
//
// Channel mixing forward:
//
//	shifted2 = TokenShift(x, mu2)
//	r2       = sigmoid(shifted2 @ Wr2)
//	k2_raw   = shifted2 @ Wk2
//	k2       = relu(k2_raw)^2
//	v2       = k2 @ Wv2
//	x        = x + r2 * v2
func emitRWKVIR(prog *Program, x string, wi, D, T, B, idx int) (int, error) {
	if D <= 0 {
		return wi, fmt.Errorf("rwkv requires D > 0, got %d", D)
	}

	prefix := tmpName(x+"_rwkv", idx)

	// --- Time mixing ---
	shifted := prefix + "_shifted"
	r := prefix + "_r"
	rSig := prefix + "_r_sig"
	k := prefix + "_k"
	v := prefix + "_v"
	expK := prefix + "_exp_k"
	kv := prefix + "_kv"
	wExp := prefix + "_w_exp"
	wNeg := prefix + "_w_neg"
	decay := prefix + "_decay"
	num := prefix + "_num"
	den := prefix + "_den"
	wkv := prefix + "_wkv"
	rWkv := prefix + "_r_wkv"
	timeOut := prefix + "_time_out"

	// Token shift: shifted = (1-sig(mu)) * x[t] + sig(mu) * x[t-1]
	emitTokenShiftIR(prog, x, weightName(wi), shifted, B, T, D, prefix+"_time")
	wi++

	// R, K, V projections from shifted input
	prog.MatMul(shifted, weightName(wi), r) // wi+1 = Wr
	wi++
	prog.Sigmoid(r, rSig)

	prog.MatMul(shifted, weightName(wi), k) // wi+2 = Wk
	wi++
	prog.MatMul(shifted, weightName(wi), v) // wi+3 = Wv
	wi++

	// exp(k) for WKV numerator and denominator
	prog.Exp(k, expK)
	prog.Mul(expK, v, kv)

	// Decay: exp(-exp(W)), keeping the tensor rank-1 so OpScan can flatten it
	// back to [D] and rely on backend broadcasting against [B, D].
	prog.Exp(weightName(wi), wExp) // wi+4 = W (decay logits)
	prog.ScalarMul(wExp, -1.0, wNeg)
	prog.Exp(wNeg, decay)

	// Gated recurrence for numerator and denominator
	prog.Scan(kv, decay, num, B, T, D)
	prog.Scan(expK, decay, den, B, T, D)
	wi++

	// WKV = num / (den + eps)
	prog.DivSafe(num, den, 1e-6, wkv)

	// Gated output: r * wkv
	prog.Mul(rSig, wkv, rWkv)

	// Output projection + residual
	prog.MatMul(rWkv, weightName(wi), timeOut) // wi+5 = Wo
	wi++
	prog.Add(x, timeOut, x)

	// --- Channel mixing ---
	shifted2 := prefix + "_shifted2"
	r2 := prefix + "_r2"
	r2Sig := prefix + "_r2_sig"
	k2Raw := prefix + "_k2_raw"
	k2Relu := prefix + "_k2_relu"
	k2 := prefix + "_k2"
	v2 := prefix + "_v2"
	chanOut := prefix + "_chan_out"

	// Token shift for channel mixing
	emitTokenShiftIR(prog, x, weightName(wi), shifted2, B, T, D, prefix+"_chan")
	wi++

	// Receptance gate: r2 = sigmoid(shifted2 @ Wr2)
	prog.MatMul(shifted2, weightName(wi), r2) // wi+7 = Wr2
	wi++
	prog.Sigmoid(r2, r2Sig)

	// Key: k2 = relu(shifted2 @ Wk2)^2
	prog.MatMul(shifted2, weightName(wi), k2Raw) // wi+8 = Wk2
	wi++
	prog.ReLU(k2Raw, k2Relu)
	prog.Square(k2Relu, k2)

	// Value: v2 = k2 @ Wv2
	prog.MatMul(k2, weightName(wi), v2) // wi+9 = Wv2
	wi++

	// Channel output + residual: x = x + r2 * v2
	prog.Mul(r2Sig, v2, chanOut)
	prog.Add(x, chanOut, x)

	return wi, nil
}

// emitPerceiverIR emits a perceiver/bottleneck block that uses cross-attention
// between a small set of learned latent tokens and the full input sequence.
//
// Forward pass:
//  1. Broadcast latent_init [L, D] -> [B*L, D]
//  2. Cross-attention: Q from latents, K/V from input x
//  3. Self-attention on latents (causal, with RoPE)
//  4. Broadcast back: Q from x, K/V from latents
//  5. Feed-forward on x (SiLU activation)
//
// Weight layout (15 weights per block):
//
//	w[wi+0]  = latent_init       [L, D]
//	w[wi+1]  = cross Q proj      [D, D]
//	w[wi+2]  = cross K proj      [D, D]
//	w[wi+3]  = cross V proj      [D, D]
//	w[wi+4]  = cross O proj      [D, D]
//	w[wi+5]  = self Q proj       [D, D]
//	w[wi+6]  = self K proj       [D, D]
//	w[wi+7]  = self V proj       [D, D]
//	w[wi+8]  = self O proj       [D, D]
//	w[wi+9]  = broad Q proj      [D, D]
//	w[wi+10] = broad K proj      [D, D]
//	w[wi+11] = broad V proj      [D, D]
//	w[wi+12] = broad O proj      [D, D]
//	w[wi+13] = FF layer 1        [D, 2*D]
//	w[wi+14] = FF layer 2        [2*D, D]
func emitPerceiverIR(prog *Program, x string, wi, H, L, D, T, B, idx int) (int, error) {
	if H <= 0 || D <= 0 || D%H != 0 {
		return wi, fmt.Errorf("invalid perceiver dimensions D=%d H=%d", D, H)
	}
	if L <= 0 {
		return wi, fmt.Errorf("perceiver requires num_latents > 0")
	}

	headDim := D / H
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	prefix := tmpName(x+"_perc", idx)
	latents := prefix + "_lat"

	// 1. Repeat latent_init [L, D] across batch -> [B*L, D]
	repeatRowsIR(prog, weightName(wi), B, latents)
	wi++

	// 2. Cross-attention: Q from latents, K/V from input
	{
		p := prefix + "_cross"
		qc := p + "_q"
		kc := p + "_k"
		vc := p + "_v"

		prog.MatMul(latents, weightName(wi), qc)
		wi++
		prog.MatMul(x, weightName(wi), kc)
		wi++
		prog.MatMul(x, weightName(wi), vc)
		wi++

		// Reshape Q: [B*L, D] -> [B, L, H, hd] -> [B, H, L, hd]
		qc4 := qc + "4"
		qch := qc + "h"
		prog.Reshape(qc, []int{B, L, H, headDim}, qc4)
		prog.Transpose(qc4, []int{0, 2, 1, 3}, qch)

		// Reshape K/V: [B*T, D] -> [B, T, H, hd] -> [B, H, T, hd]
		kc4 := kc + "4"
		kch := kc + "h"
		vc4 := vc + "4"
		vch := vc + "h"
		prog.Reshape(kc, []int{B, T, H, headDim}, kc4)
		prog.Transpose(kc4, []int{0, 2, 1, 3}, kch)
		prog.Reshape(vc, []int{B, T, H, headDim}, vc4)
		prog.Transpose(vc4, []int{0, 2, 1, 3}, vch)

		// Scores: Q @ K^T / sqrt(headDim), no causal mask
		kct := kc + "t"
		scores := p + "_scores"
		scaled := scores + "_scaled"
		attn := p + "_attn"
		ctx := p + "_ctx"
		ctxT := p + "_ctx_t"
		flat := p + "_flat"
		proj := p + "_proj"

		prog.Transpose(kch, []int{0, 1, 3, 2}, kct)
		prog.MatMul(qch, kct, scores)
		prog.ScalarMul(scores, scale, scaled)
		prog.Softmax(scaled, -1, attn)

		prog.MatMul(attn, vch, ctx)
		prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
		prog.Reshape(ctxT, []int{B * L, D}, flat)

		prog.MatMul(flat, weightName(wi), proj)
		wi++
		prog.Add(latents, proj, latents)
	}

	// 3. Self-attention on latents (with causal mask + RoPE)
	{
		p := prefix + "_self"
		qs := p + "_q"
		ks := p + "_k"
		vs := p + "_v"

		prog.MatMul(latents, weightName(wi), qs)
		wi++
		prog.MatMul(latents, weightName(wi), ks)
		wi++
		prog.MatMul(latents, weightName(wi), vs)
		wi++

		// Reshape: [B*L, D] -> [B, L, H, hd] -> [B, H, L, hd]
		qs4 := qs + "4"
		qsh := qs + "h"
		ks4 := ks + "4"
		ksh := ks + "h"
		vs4 := vs + "4"
		vsh := vs + "h"
		prog.Reshape(qs, []int{B, L, H, headDim}, qs4)
		prog.Transpose(qs4, []int{0, 2, 1, 3}, qsh)
		prog.Reshape(ks, []int{B, L, H, headDim}, ks4)
		prog.Transpose(ks4, []int{0, 2, 1, 3}, ksh)
		prog.Reshape(vs, []int{B, L, H, headDim}, vs4)
		prog.Transpose(vs4, []int{0, 2, 1, 3}, vsh)

		// RoPE
		qsRot := qs + "_rot"
		ksRot := ks + "_rot"
		prog.RoPE(qsh, ksh, qsRot, ksRot, L, headDim, 10000.0)

		// Scores with causal mask
		kst := ks + "t"
		scores := p + "_scores"
		scaled := scores + "_scaled"
		masked := scores + "_masked"
		attn := p + "_attn"
		ctx := p + "_ctx"
		ctxT := p + "_ctx_t"
		flat := p + "_flat"
		proj := p + "_proj"

		prog.Transpose(ksRot, []int{0, 1, 3, 2}, kst)
		prog.MatMul(qsRot, kst, scores)
		prog.ScalarMul(scores, scale, scaled)
		prog.CausalMask(scaled, L, masked)
		prog.Softmax(masked, -1, attn)

		prog.MatMul(attn, vsh, ctx)
		prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
		prog.Reshape(ctxT, []int{B * L, D}, flat)

		prog.MatMul(flat, weightName(wi), proj)
		wi++
		prog.Add(latents, proj, latents)
	}

	// 4. Broadcast back: Q from x, K/V from latents
	{
		p := prefix + "_broad"
		qb := p + "_q"
		kb := p + "_k"
		vb := p + "_v"

		prog.MatMul(x, weightName(wi), qb)
		wi++
		prog.MatMul(latents, weightName(wi), kb)
		wi++
		prog.MatMul(latents, weightName(wi), vb)
		wi++

		// Reshape Q: [B*T, D] -> [B, T, H, hd] -> [B, H, T, hd]
		qb4 := qb + "4"
		qbh := qb + "h"
		prog.Reshape(qb, []int{B, T, H, headDim}, qb4)
		prog.Transpose(qb4, []int{0, 2, 1, 3}, qbh)

		// Reshape K/V: [B*L, D] -> [B, L, H, hd] -> [B, H, L, hd]
		kb4 := kb + "4"
		kbh := kb + "h"
		vb4 := vb + "4"
		vbh := vb + "h"
		prog.Reshape(kb, []int{B, L, H, headDim}, kb4)
		prog.Transpose(kb4, []int{0, 2, 1, 3}, kbh)
		prog.Reshape(vb, []int{B, L, H, headDim}, vb4)
		prog.Transpose(vb4, []int{0, 2, 1, 3}, vbh)

		// Scores: Q @ K^T / sqrt(headDim), no causal mask
		kbt := kb + "t"
		scores := p + "_scores"
		scaled := scores + "_scaled"
		attn := p + "_attn"
		ctx := p + "_ctx"
		ctxT := p + "_ctx_t"
		flat := p + "_flat"
		proj := p + "_proj"

		prog.Transpose(kbh, []int{0, 1, 3, 2}, kbt)
		prog.MatMul(qbh, kbt, scores)
		prog.ScalarMul(scores, scale, scaled)
		prog.Softmax(scaled, -1, attn)

		prog.MatMul(attn, vbh, ctx)
		prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
		prog.Reshape(ctxT, []int{B * T, D}, flat)

		prog.MatMul(flat, weightName(wi), proj)
		wi++
		prog.Add(x, proj, x)
	}

	// 5. Feed-forward on x: ff1 -> SiLU -> ff2 -> residual
	ff1 := prefix + "_ff1"
	ffAct := prefix + "_ff_act"
	ff2 := prefix + "_ff2"
	prog.MatMul(x, weightName(wi), ff1)
	wi++
	prog.SiLU(ff1, ffAct)
	prog.MatMul(ffAct, weightName(wi), ff2)
	wi++
	prog.Add(x, ff2, x)

	return wi, nil
}

// emitRetNetIR emits a RetNet retention block with multi-scale exponential
// decay, replacing the causal mask + softmax with a learned decay mask.
//
// Weight layout (8 weights per block):
//
//	w[wi+0] = RMSNorm scale
//	w[wi+1] = Q projection  [D, D]
//	w[wi+2] = K projection  [D, D]
//	w[wi+3] = V projection  [D, D]
//	w[wi+4] = decay logits  [H]  (passed through sigmoid to get rate in (0,1))
//	w[wi+5] = output projection [D, D]
//	w[wi+6] = FF layer 1    [D, 2*D]
//	w[wi+7] = FF layer 2    [2*D, D]
//
// Forward pass:
//
//	xNorm = rmsnorm(x)
//	Q, K, V = xNorm @ Wq, xNorm @ Wk, xNorm @ Wv
//	reshape to [B, H, T, headDim]
//	scores = Q @ K^T / sqrt(headDim)
//	attn = retention(scores, decay)  -- exp-decay causal mask + row normalize
//	ctx = attn @ V
//	proj = flatten(ctx) @ Wo
//	x = x + proj
//	ff1 = x @ Wff1; ff_act = silu(ff1); ff2 = ff_act @ Wff2
//	x = x + ff2
func emitRetNetIR(prog *Program, x string, wi, H, D, T, B, idx int) (int, error) {
	if H <= 0 || D <= 0 || D%H != 0 {
		return wi, fmt.Errorf("invalid retnet dimensions D=%d H=%d", D, H)
	}
	headDim := D / H
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	prefix := tmpName(x+"_retnet", idx)
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
	scores := prefix + "_scores"
	scaled := scores + "_scaled"
	attn := prefix + "_attn"
	ctx := prefix + "_ctx"
	ctxT := prefix + "_ctx_t"
	flat := prefix + "_flat"
	proj := prefix + "_proj"

	// Pre-attention RMSNorm
	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	// Q/K/V projections
	prog.MatMul(xNorm, weightName(wi), q)
	wi++
	prog.MatMul(xNorm, weightName(wi), k)
	wi++
	prog.MatMul(xNorm, weightName(wi), v)
	wi++

	// Reshape to multi-head: [B*T, D] -> [B, T, H, headDim] -> [B, H, T, headDim]
	prog.Reshape(q, []int{B, T, H, headDim}, q4)
	prog.Transpose(q4, []int{0, 2, 1, 3}, qh)
	prog.Reshape(k, []int{B, T, H, headDim}, k4)
	prog.Transpose(k4, []int{0, 2, 1, 3}, kh)
	prog.Reshape(v, []int{B, T, H, headDim}, v4)
	prog.Transpose(v4, []int{0, 2, 1, 3}, vh)

	// Attention scores: Q @ K^T / sqrt(headDim)
	prog.Transpose(kh, []int{0, 1, 3, 2}, kt)
	prog.MatMul(qh, kt, scores)
	prog.ScalarMul(scores, scale, scaled)

	// Retention approximation built from existing ops:
	// exp(scores) * exp(-sigmoid(decay_h) * (i-j)) for j<=i, then row-normalize.
	decaySig := prefix + "_decay_sig"
	pos := prefix + "_pos"
	ones := prefix + "_ones"
	rowPos := prefix + "_row_pos"
	colPos := prefix + "_col_pos"
	delta := prefix + "_delta"
	delta4 := prefix + "_delta4"
	decay4 := prefix + "_decay4"
	decayScaled := prefix + "_decay_scaled"
	decayNeg := prefix + "_decay_neg"
	decayMasked := prefix + "_decay_masked"
	decayWeights := prefix + "_decay_weights"
	expScores := prefix + "_exp_scores"
	weighted := prefix + "_weighted"
	rowMean := prefix + "_row_mean"
	rowSum := prefix + "_row_sum"
	rowSum4 := prefix + "_row_sum4"

	prog.Sigmoid(weightName(wi), decaySig)
	prog.Arange(0, T, pos)
	prog.Full([]int{T}, 1.0, ones)
	prog.Outer(pos, ones, rowPos)
	prog.Outer(ones, pos, colPos)
	prog.Sub(rowPos, colPos, delta)
	prog.Reshape(delta, []int{1, 1, T, T}, delta4)
	prog.Reshape(decaySig, []int{1, H, 1, 1}, decay4)
	prog.Mul(decay4, delta4, decayScaled)
	prog.ScalarMul(decayScaled, -1.0, decayNeg)
	prog.CausalMask(decayNeg, T, decayMasked)
	prog.Exp(decayMasked, decayWeights)

	prog.Exp(scaled, expScores)
	prog.Mul(expScores, decayWeights, weighted)
	prog.MeanAxis(weighted, 3, rowMean)
	prog.ScalarMul(rowMean, float32(T), rowSum)
	prog.Reshape(rowSum, []int{B, H, T, 1}, rowSum4)
	prog.DivSafe(weighted, rowSum4, 1e-6, attn)
	wi++

	// Attention output: attn @ V, then transpose back
	prog.MatMul(attn, vh, ctx)
	prog.Transpose(ctx, []int{0, 2, 1, 3}, ctxT)
	prog.Reshape(ctxT, []int{B * T, D}, flat)

	// Output projection + residual
	prog.MatMul(flat, weightName(wi), proj)
	wi++
	prog.Add(x, proj, x)

	// Feed-forward tail: ff1 -> SiLU -> ff2 -> residual
	retFF1 := prefix + "_ff1"
	retFFAct := prefix + "_ff_act"
	retFF2 := prefix + "_ff2"
	prog.MatMul(x, weightName(wi), retFF1)
	wi++
	prog.SiLU(retFF1, retFFAct)
	prog.MatMul(retFFAct, weightName(wi), retFF2)
	wi++
	prog.Add(x, retFF2, x)

	return wi, nil
}
