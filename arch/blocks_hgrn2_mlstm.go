package arch

import "fmt"

func effectiveHGRN2DState(spec BlockSpec, D int) (int, error) {
	if spec.Heads <= 0 {
		return 0, fmt.Errorf("hgrn2 requires heads > 0")
	}
	if D <= 0 || D%spec.Heads != 0 {
		return 0, fmt.Errorf("hgrn2 requires model_dim divisible by heads (model_dim=%d heads=%d)", D, spec.Heads)
	}
	if spec.DState > 0 {
		return spec.DState, nil
	}
	return D / spec.Heads, nil
}

func hgrn2WeightCount(_ BlockSpec, _, _ bool) (int, error) {
	return 6, nil
}

func hgrn2WeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	ds, err := effectiveHGRN2DState(spec, D)
	if err != nil {
		return nil, err
	}
	headDim := D / spec.Heads
	stateDim := spec.Heads * ds
	return []WeightMeta{
		{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "w_v", Shape: []int{D, D}, InitMode: "torch_linear_uniform"},
		{Name: "w_q", Shape: []int{D, stateDim}, InitMode: "torch_linear_uniform"},
		{Name: "w_f", Shape: []int{D, stateDim}, InitMode: "torch_linear_uniform"},
		{Name: "o_norm_scale", Shape: []int{headDim}, IsNormScale: true, InitOne: true},
		{Name: "wo", Shape: []int{D, D}, InitMode: "torch_linear_uniform"},
	}, nil
}

func emitHGRN2IR(prog *Program, spec BlockSpec, x string, wi, D, T, B, idx int) (int, error) {
	heads := spec.Heads
	ds, err := effectiveHGRN2DState(spec, D)
	if err != nil {
		return wi, err
	}
	headDim := D / heads
	stateDim := heads * ds

	prefix := tmpName(x+"_hgrn2", idx)
	xNorm := prefix + "_x_norm"
	vRaw := prefix + "_v_raw"
	vAct := prefix + "_v_act"
	v4 := prefix + "_v4"
	qRaw := prefix + "_q_raw"
	qFlat := prefix + "_q_flat"
	q4 := prefix + "_q4"
	forgetRaw := prefix + "_forget_raw"
	forgetFlat := prefix + "_forget_flat"
	forget4 := prefix + "_forget4"
	inputOnes := prefix + "_input_ones"
	kFlat := prefix + "_k_flat"
	k4 := prefix + "_k4"
	yFlat := prefix + "_y_flat"
	yNorm := prefix + "_y_norm"
	yMerged := prefix + "_y_merged"
	out := prefix + "_out"

	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	prog.MatMul(xNorm, weightName(wi), vRaw)
	wi++
	prog.SiLU(vRaw, vAct)
	prog.Reshape(vAct, []int{B, T, heads, headDim}, v4)

	prog.MatMul(xNorm, weightName(wi), qRaw)
	wi++
	prog.Sigmoid(qRaw, qFlat)
	prog.Reshape(qFlat, []int{B, T, heads, ds}, q4)

	prog.MatMul(xNorm, weightName(wi), forgetRaw)
	wi++
	prog.Sigmoid(forgetRaw, forgetFlat)
	prog.Reshape(forgetFlat, []int{B, T, heads, ds}, forget4)
	prog.Full([]int{B * T, stateDim}, 1.0, inputOnes)
	prog.Sub(inputOnes, forgetFlat, kFlat)
	prog.Reshape(kFlat, []int{B, T, heads, ds}, k4)

	prog.HGRN2Scan(q4, k4, v4, forget4, yFlat, B, T, heads, ds, headDim)
	prog.RMSNorm(yFlat, weightName(wi), yNorm, 1e-5)
	wi++
	prog.Reshape(yNorm, []int{B * T, D}, yMerged)
	prog.MatMul(yMerged, weightName(wi), out)
	wi++
	prog.Add(x, out, x)
	return wi, nil
}

func mlstmWeightCount(_ BlockSpec, _, _ bool) (int, error) {
	return 11, nil
}

func mlstmWeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	if spec.Heads <= 0 {
		return nil, fmt.Errorf("mlstm requires heads > 0")
	}
	if spec.DK <= 0 {
		return nil, fmt.Errorf("mlstm requires d_k > 0")
	}
	if spec.DV <= 0 {
		return nil, fmt.Errorf("mlstm requires d_v > 0")
	}
	keyDim := spec.Heads * spec.DK
	valDim := spec.Heads * spec.DV
	return []WeightMeta{
		{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "wq", Shape: []int{D, keyDim}, InitMode: "torch_linear_uniform"},
		{Name: "wk", Shape: []int{D, keyDim}, InitMode: "torch_linear_uniform"},
		{Name: "wv", Shape: []int{D, valDim}, InitMode: "torch_linear_uniform"},
		{Name: "w_i", Shape: []int{D, spec.Heads}, InitMode: "torch_linear_uniform"},
		{Name: "b_i", Shape: []int{spec.Heads}},
		{Name: "w_f", Shape: []int{D, spec.Heads}, InitMode: "torch_linear_uniform"},
		{Name: "b_f", Shape: []int{spec.Heads}, InitOne: true},
		{Name: "o_norm_scale", Shape: []int{spec.DV}, IsNormScale: true, InitOne: true},
		{Name: "w_out_gate", Shape: []int{D, valDim}, InitMode: "torch_linear_uniform"},
		{Name: "wo", Shape: []int{valDim, D}, InitMode: "torch_linear_uniform"},
	}, nil
}

func emitMLSTMIR(prog *Program, spec BlockSpec, x string, wi, T, B, idx int) (int, error) {
	if spec.Heads <= 0 {
		return wi, fmt.Errorf("mlstm requires heads > 0")
	}
	if spec.DK <= 0 {
		return wi, fmt.Errorf("mlstm requires d_k > 0")
	}
	if spec.DV <= 0 {
		return wi, fmt.Errorf("mlstm requires d_v > 0")
	}
	heads := spec.Heads
	dk := spec.DK
	dv := spec.DV
	valDim := heads * dv

	prefix := tmpName(x+"_mlstm", idx)
	xNorm := prefix + "_x_norm"
	qFlat := prefix + "_q_flat"
	kFlat := prefix + "_k_flat"
	vFlat := prefix + "_v_flat"
	q4 := prefix + "_q4"
	k4 := prefix + "_k4"
	v4 := prefix + "_v4"
	inputRaw := prefix + "_input_raw"
	inputBias := prefix + "_input_bias"
	input3 := prefix + "_input3"
	forgetRaw := prefix + "_forget_raw"
	forgetBias := prefix + "_forget_bias"
	forget3 := prefix + "_forget3"
	yFlat := prefix + "_y_flat"
	yNorm := prefix + "_y_norm"
	outGateRaw := prefix + "_out_gate_raw"
	outGate4 := prefix + "_out_gate4"
	outGateFlat := prefix + "_out_gate_flat"
	outGate := prefix + "_out_gate"
	yGated := prefix + "_y_gated"
	yMerged := prefix + "_y_merged"
	out := prefix + "_out"

	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++

	prog.MatMul(xNorm, weightName(wi), qFlat)
	wi++
	prog.Reshape(qFlat, []int{B, T, heads, dk}, q4)

	prog.MatMul(xNorm, weightName(wi), kFlat)
	wi++
	prog.Reshape(kFlat, []int{B, T, heads, dk}, k4)

	prog.MatMul(xNorm, weightName(wi), vFlat)
	wi++
	prog.Reshape(vFlat, []int{B, T, heads, dv}, v4)

	prog.MatMul(xNorm, weightName(wi), inputRaw)
	wi++
	prog.Add(inputRaw, weightName(wi), inputBias)
	wi++
	prog.Reshape(inputBias, []int{B, T, heads}, input3)

	prog.MatMul(xNorm, weightName(wi), forgetRaw)
	wi++
	prog.Add(forgetRaw, weightName(wi), forgetBias)
	wi++
	prog.Reshape(forgetBias, []int{B, T, heads}, forget3)

	prog.MLSTMScan(q4, k4, v4, input3, forget3, yFlat, B, T, heads, dk, dv)
	prog.RMSNorm(yFlat, weightName(wi), yNorm, 1e-5)
	wi++

	prog.MatMul(xNorm, weightName(wi), outGateRaw)
	wi++
	prog.Reshape(outGateRaw, []int{B, T, heads, dv}, outGate4)
	prog.Reshape(outGate4, []int{B * T * heads, dv}, outGateFlat)
	prog.Sigmoid(outGateFlat, outGate)
	prog.Mul(yNorm, outGate, yGated)
	prog.Reshape(yGated, []int{B * T, valDim}, yMerged)

	prog.MatMul(yMerged, weightName(wi), out)
	wi++
	prog.Add(x, out, x)
	return wi, nil
}

func estimateHGRN2BlockFLOPs(block BlockSpec, B, T, D int) int64 {
	if block.Heads <= 0 || D <= 0 || D%block.Heads != 0 {
		return 0
	}
	ds, err := effectiveHGRN2DState(block, D)
	if err != nil {
		return 0
	}
	headDim := D / block.Heads
	stateDim := block.Heads * ds
	proj := 2*i64(B)*i64(T)*i64(D)*i64(D) + 2*2*i64(B)*i64(T)*i64(D)*i64(stateDim)
	scan := 4 * i64(B) * i64(T) * i64(block.Heads) * i64(ds) * i64(headDim)
	out := 2 * i64(B) * i64(T) * i64(D) * i64(D)
	return proj + scan + out
}

func estimateMLSTMBlockFLOPs(block BlockSpec, B, T, D int) int64 {
	if block.Heads <= 0 || block.DK <= 0 || block.DV <= 0 {
		return 0
	}
	keyDim := block.Heads * block.DK
	valDim := block.Heads * block.DV
	proj := 2*i64(B)*i64(T)*i64(D)*i64(keyDim)*2 +
		2*i64(B)*i64(T)*i64(D)*i64(valDim) +
		2*i64(B)*i64(T)*i64(D)*i64(block.Heads)*2 +
		2*i64(B)*i64(T)*i64(D)*i64(valDim) +
		2*i64(B)*i64(T)*i64(valDim)*i64(D)
	scan := 6 * i64(B) * i64(T) * i64(block.Heads) * i64(block.DK) * i64(block.DV)
	elementwise := i64(B) * i64(T) * i64(block.Heads) * i64(block.DV)
	return proj + scan + elementwise
}
