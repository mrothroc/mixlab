package arch

import (
	"fmt"
	"math"
)

const gatedDeltaNetConvSize = 4

func effectiveGatedDeltaNetDV(spec BlockSpec) int {
	if spec.DV > 0 {
		return spec.DV
	}
	return 2 * spec.DK
}

func effectiveKVShare(spec BlockSpec) bool {
	if spec.KVShare == nil {
		return true
	}
	return *spec.KVShare
}

func gatedDeltaNetWeightCount(spec BlockSpec, _, _ bool) (int, error) {
	if effectiveKVShare(spec) {
		return 13, nil
	}
	return 14, nil
}

func gatedDeltaNetWeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	if spec.Heads <= 0 {
		return nil, fmt.Errorf("gated_deltanet requires heads > 0")
	}
	if spec.DK <= 0 {
		return nil, fmt.Errorf("gated_deltanet requires d_k > 0")
	}
	dv := effectiveGatedDeltaNetDV(spec)
	if dv <= 0 {
		return nil, fmt.Errorf("gated_deltanet requires d_v > 0")
	}
	if effectiveKVShare(spec) && dv < spec.DK {
		return nil, fmt.Errorf("gated_deltanet with kv_share=true requires d_v >= d_k")
	}

	keyDim := spec.Heads * spec.DK
	valDim := spec.Heads * dv
	metas := []WeightMeta{
		{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "wq", Shape: []int{D, keyDim}, InitMode: "torch_linear_uniform"},
		{Name: "q_conv", Shape: []int{gatedDeltaNetConvSize, keyDim}, InitMode: "torch_depthwise_conv1d_uniform"},
	}
	if effectiveKVShare(spec) {
		metas = append(metas, WeightMeta{Name: "w_kv", Shape: []int{D, valDim}, InitMode: "torch_linear_uniform"})
		metas = append(metas,
			WeightMeta{Name: "k_conv", Shape: []int{gatedDeltaNetConvSize, keyDim}, InitMode: "torch_depthwise_conv1d_uniform"},
			WeightMeta{Name: "v_conv", Shape: []int{gatedDeltaNetConvSize, valDim}, InitMode: "torch_depthwise_conv1d_uniform"},
		)
	} else {
		metas = append(metas,
			WeightMeta{Name: "wk", Shape: []int{D, keyDim}, InitMode: "torch_linear_uniform"},
			WeightMeta{Name: "k_conv", Shape: []int{gatedDeltaNetConvSize, keyDim}, InitMode: "torch_depthwise_conv1d_uniform"},
			WeightMeta{Name: "wv", Shape: []int{D, valDim}, InitMode: "torch_linear_uniform"},
			WeightMeta{Name: "v_conv", Shape: []int{gatedDeltaNetConvSize, valDim}, InitMode: "torch_depthwise_conv1d_uniform"},
		)
	}
	metas = append(metas,
		WeightMeta{Name: "w_a", Shape: []int{D, spec.Heads}, InitMode: "torch_linear_uniform"},
		WeightMeta{Name: "A_log", Shape: []int{spec.Heads}, InitMode: "gated_deltanet_A_log"},
		WeightMeta{Name: "dt_bias", Shape: []int{spec.Heads}, InitMode: "gated_deltanet_dt_bias"},
		WeightMeta{Name: "w_beta", Shape: []int{D, spec.Heads}, InitMode: "torch_linear_uniform"},
		WeightMeta{Name: "o_norm_scale", Shape: []int{dv}, IsNormScale: true, InitOne: true},
		WeightMeta{Name: "w_out_gate", Shape: []int{D, valDim}, InitMode: "torch_linear_uniform"},
		WeightMeta{Name: "wo", Shape: []int{valDim, D}, InitMode: "torch_linear_uniform"},
	)
	return metas, nil
}

func emitGatedDeltaNetIR(prog *Program, spec BlockSpec, x string, wi, _, T, B, idx int) (int, error) {
	heads := spec.Heads
	if heads <= 0 {
		return wi, fmt.Errorf("gated_deltanet requires heads > 0")
	}
	dk := spec.DK
	if dk <= 0 {
		return wi, fmt.Errorf("gated_deltanet requires d_k > 0")
	}
	dv := effectiveGatedDeltaNetDV(spec)
	if dv <= 0 {
		return wi, fmt.Errorf("gated_deltanet requires d_v > 0")
	}
	if effectiveKVShare(spec) && dv < dk {
		return wi, fmt.Errorf("gated_deltanet with kv_share=true requires d_v >= d_k")
	}

	keyDim := heads * dk
	valDim := heads * dv
	prefix := tmpName(x+"_gated_deltanet", idx)
	xNorm := prefix + "_x_norm"
	qProj := prefix + "_q_proj"
	qSeq := prefix + "_q_seq"
	kProj := prefix + "_k_proj"
	kSeq := prefix + "_k_seq"
	kvWide := prefix + "_kv_wide"
	kvSeq := prefix + "_kv_seq"
	vProj := prefix + "_v_proj"
	vSeq := prefix + "_v_seq"
	q4 := prefix + "_q4"
	k4 := prefix + "_k4"
	v4 := prefix + "_v4"
	qFlat := prefix + "_q_flat"
	kFlat := prefix + "_k_flat"
	qOnes := prefix + "_q_ones"
	qUnit := prefix + "_q_unit"
	kUnit := prefix + "_k_unit"
	qL2 := prefix + "_q_l2"
	kL2 := prefix + "_k_l2"
	qScaled := prefix + "_q_scaled"
	qScaled4 := prefix + "_q_scaled4"
	gateRaw := prefix + "_gate_raw"
	dtShifted := prefix + "_dt_shifted"
	dt := prefix + "_dt"
	aScale := prefix + "_a_scale"
	decayRate := prefix + "_decay_rate"
	decayLogits := prefix + "_decay_logits"
	gateHead := prefix + "_gate_head"
	betaRaw := prefix + "_beta_raw"
	betaHead := prefix + "_beta_head"
	yFlat := prefix + "_y_flat"
	yNorm := prefix + "_y_norm"
	outGateRaw := prefix + "_out_gate_raw"
	outGate := prefix + "_out_gate"
	outGate4 := prefix + "_out_gate4"
	outGateFlat := prefix + "_out_gate_flat"
	yGated := prefix + "_y_gated"
	yMerged := prefix + "_y_merged"
	out := prefix + "_out"

	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	prog.MatMul(xNorm, weightName(wi), qProj)
	wi++
	prog.Reshape(qProj, []int{B, T, keyDim}, qProj)
	emitGatedDeltaNetShortConv1D(prog, qProj, weightName(wi), B, T, keyDim, prefix+"_qconv", qSeq)
	wi++

	if effectiveKVShare(spec) {
		prog.MatMul(xNorm, weightName(wi), kvWide)
		wi++
		prog.Reshape(kvWide, []int{B, T, heads, dv}, kvWide)
		prog.Slice(kvWide, 0, dk, 1, 3, k4)
		prog.Reshape(k4, []int{B, T, keyDim}, kProj)
		prog.Reshape(kvWide, []int{B, T, valDim}, kvSeq)
		emitGatedDeltaNetShortConv1D(prog, kProj, weightName(wi), B, T, keyDim, prefix+"_kconv", kSeq)
		wi++
		emitGatedDeltaNetShortConv1D(prog, kvSeq, weightName(wi), B, T, valDim, prefix+"_vconv", vSeq)
		wi++
	} else {
		prog.MatMul(xNorm, weightName(wi), kProj)
		wi++
		prog.Reshape(kProj, []int{B, T, keyDim}, kProj)
		emitGatedDeltaNetShortConv1D(prog, kProj, weightName(wi), B, T, keyDim, prefix+"_kconv", kSeq)
		wi++
		prog.MatMul(xNorm, weightName(wi), vProj)
		wi++
		prog.Reshape(vProj, []int{B, T, valDim}, vProj)
		emitGatedDeltaNetShortConv1D(prog, vProj, weightName(wi), B, T, valDim, prefix+"_vconv", vSeq)
		wi++
	}

	prog.Reshape(qSeq, []int{B, T, heads, dk}, q4)
	prog.Reshape(kSeq, []int{B, T, heads, dk}, k4)
	prog.Reshape(vSeq, []int{B, T, heads, dv}, v4)

	prog.Reshape(q4, []int{B * T * heads, dk}, qFlat)
	prog.Reshape(k4, []int{B * T * heads, dk}, kFlat)
	prog.Full([]int{dk}, 1.0, qOnes)
	prog.RMSNorm(qFlat, qOnes, qUnit, 1e-6)
	prog.RMSNorm(kFlat, qOnes, kUnit, 1e-6)
	prog.ScalarMul(qUnit, float32(1.0/math.Sqrt(float64(dk))), qL2)
	prog.ScalarMul(kUnit, float32(1.0/math.Sqrt(float64(dk))), kL2)
	prog.ScalarMul(qL2, float32(1.0/math.Sqrt(float64(dk))), qScaled)
	prog.Reshape(qScaled, []int{B, T, heads, dk}, qScaled4)
	prog.Reshape(kL2, []int{B, T, heads, dk}, k4)

	prog.MatMul(xNorm, weightName(wi), gateRaw)
	wi++
	prog.Add(gateRaw, weightName(wi+1), dtShifted)
	prog.Softplus(dtShifted, dt)
	prog.Exp(weightName(wi), aScale)
	prog.Mul(dt, aScale, decayRate)
	prog.ScalarMul(decayRate, -1.0, decayLogits)
	prog.Exp(decayLogits, gateHead)
	wi += 2

	prog.MatMul(xNorm, weightName(wi), betaRaw)
	wi++
	prog.Sigmoid(betaRaw, betaHead)

	prog.GatedDeltaScan(qScaled4, k4, v4, betaHead, gateHead, yFlat, B, T, heads, dk, dv)
	prog.RMSNorm(yFlat, weightName(wi), yNorm, 1e-5)
	wi++

	prog.MatMul(xNorm, weightName(wi), outGateRaw)
	wi++
	prog.Reshape(outGateRaw, []int{B, T, heads, dv}, outGate4)
	prog.Reshape(outGate4, []int{B * T * heads, dv}, outGateFlat)
	prog.SiLU(outGateFlat, outGate)
	prog.Mul(yNorm, outGate, yGated)
	prog.Reshape(yGated, []int{B * T, valDim}, yMerged)

	prog.MatMul(yMerged, weightName(wi), out)
	wi++
	prog.Add(x, out, x)

	return wi, nil
}

func emitGatedDeltaNetShortConv1D(prog *Program, seq3D, weight string, B, T, C int, prefix, out string) {
	acc := prefix + "_acc"
	prog.Full([]int{B, T, C}, 0.0, acc)
	for tap := 0; tap < gatedDeltaNetConvSize; tap++ {
		shifted := prefix + fmt.Sprintf("_shift_%d", tap)
		if tap == 0 {
			prog.ScalarMul(seq3D, 1.0, shifted)
		} else {
			zeros := prefix + fmt.Sprintf("_pad_%d", tap)
			trimmed := prefix + fmt.Sprintf("_trim_%d", tap)
			prog.Full([]int{B, tap, C}, 0.0, zeros)
			prog.Slice(seq3D, 0, T-tap, 1, 1, trimmed)
			prog.Concat(zeros, trimmed, 1, shifted)
		}
		row := prefix + fmt.Sprintf("_wrow_%d", tap)
		row3D := prefix + fmt.Sprintf("_wrow3d_%d", tap)
		term := prefix + fmt.Sprintf("_term_%d", tap)
		prog.Slice(weight, tap, tap+1, 1, 0, row)
		prog.Reshape(row, []int{1, 1, C}, row3D)
		prog.Mul(shifted, row3D, term)
		prog.Add(acc, term, acc)
	}
	prog.SiLU(acc, out)
}
