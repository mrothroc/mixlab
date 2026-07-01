package arch

import (
	"fmt"
	"strings"
)

const (
	moeExpertSwiGLU = iota
	moeExpertGEGLU
	moeExpertMLP
)

const (
	moeActivationSiLU = iota
	moeActivationGELU
	moeActivationReLU
	moeActivationLeakyReLUSq
)

func effectiveMoEExpertBlock(spec BlockSpec) BlockSpec {
	if spec.ExpertBlock == nil {
		return BlockSpec{Type: "swiglu"}
	}
	return *spec.ExpertBlock
}

func effectiveMoETopK(spec BlockSpec) int {
	if spec.TopK > 0 {
		return spec.TopK
	}
	if spec.NumExperts <= 1 {
		return 1
	}
	return 2
}

func effectiveMoELoadBalanceLossWeight(spec BlockSpec) float64 {
	if !spec.loadBalanceLossWeightSet && spec.LoadBalanceLossWeight == 0 {
		return 0.01
	}
	return spec.LoadBalanceLossWeight
}

func moeRouter(spec BlockSpec) string {
	router := strings.ToLower(strings.TrimSpace(spec.Router))
	if router == "" {
		return "linear"
	}
	return router
}

func moeExpertTypeAndActivation(spec BlockSpec) (int, int, float32, error) {
	expert := effectiveMoEExpertBlock(spec)
	switch blockTypeKey(expert) {
	case "swiglu":
		return moeExpertSwiGLU, moeActivationSiLU, 0, nil
	case "geglu":
		return moeExpertGEGLU, moeActivationGELU, 0, nil
	case "mlp":
		switch strings.ToLower(strings.TrimSpace(expert.Activation)) {
		case "", "silu":
			return moeExpertMLP, moeActivationSiLU, 0, nil
		case "gelu":
			return moeExpertMLP, moeActivationGELU, 0, nil
		case "relu":
			return moeExpertMLP, moeActivationReLU, 0, nil
		case "leaky_relu_sq":
			slope := expert.LeakySlope
			if slope == 0 {
				slope = 0.5
			}
			return moeExpertMLP, moeActivationLeakyReLUSq, float32(slope), nil
		default:
			return 0, 0, 0, fmt.Errorf("unsupported moe mlp activation %q", expert.Activation)
		}
	default:
		return 0, 0, 0, fmt.Errorf("unsupported moe expert block type %q", expert.Type)
	}
}

func moeExpertWeightCount(spec BlockSpec) (int, error) {
	expertType, _, _, err := moeExpertTypeAndActivation(spec)
	if err != nil {
		return 0, err
	}
	switch expertType {
	case moeExpertSwiGLU, moeExpertGEGLU:
		return 3, nil
	case moeExpertMLP:
		return 2, nil
	default:
		return 0, fmt.Errorf("unsupported moe expert type %d", expertType)
	}
}

func moeWeightCount(spec BlockSpec, blockScales, _ bool) (int, error) {
	if spec.NumExperts <= 0 {
		return 0, fmt.Errorf("moe requires num_experts > 0")
	}
	perExpert, err := moeExpertWeightCount(spec)
	if err != nil {
		return 0, err
	}
	total := 2 + spec.NumExperts*perExpert // norm + router + expert weights
	if blockScales {
		total++
	}
	return total, nil
}

func moeWeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	return moeWeightShapesWithOptions(spec, D, DefaultFFNMultiplier, false)
}

func moeWeightShapesWithOptions(spec BlockSpec, D int, mlpMult float64, blockScales bool) ([]WeightMeta, error) {
	if spec.NumExperts <= 0 {
		return nil, fmt.Errorf("moe requires num_experts > 0")
	}
	expertType, _, _, err := moeExpertTypeAndActivation(spec)
	if err != nil {
		return nil, err
	}
	ffn := ffnDim(D, mlpMult)
	metas := []WeightMeta{
		{Name: "moe_norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
		{Name: "router_w", Shape: []int{D, spec.NumExperts}},
	}
	for i := 0; i < spec.NumExperts; i++ {
		switch expertType {
		case moeExpertSwiGLU, moeExpertGEGLU:
			metas = append(metas,
				WeightMeta{Name: fmt.Sprintf("expert_%d_w_gate", i), Shape: []int{D, ffn}},
				WeightMeta{Name: fmt.Sprintf("expert_%d_w_up", i), Shape: []int{D, ffn}},
				WeightMeta{Name: fmt.Sprintf("expert_%d_w_down", i), Shape: []int{ffn, D}},
			)
		case moeExpertMLP:
			metas = append(metas,
				WeightMeta{Name: fmt.Sprintf("expert_%d_w_up", i), Shape: []int{D, ffn}},
				WeightMeta{Name: fmt.Sprintf("expert_%d_w_down", i), Shape: []int{ffn, D}},
			)
		default:
			return nil, fmt.Errorf("unsupported moe expert type %d", expertType)
		}
	}
	if blockScales {
		metas = append(metas, WeightMeta{Name: "moe_scale", Shape: []int{D}, InitOne: true})
	}
	return metas, nil
}

func emitMoEIR(prog *Program, spec BlockSpec, x string, wi, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, layerAgg *layerAggregationBuildState, adaLN *adaLNBuildState, blockIndex int) (int, error) {
	prefix := tmpName(x+"_moe", idx)
	xNorm := prefix + "_x_norm"
	prog.RMSNorm(x, weightName(wi), xNorm, 1e-5)
	wi++
	xNorm = adaLN.apply(prog, xNorm, B, T, blockIndex, "moe")

	delta, nextWI, err := emitMoEDeltaIR(prog, spec, xNorm, wi, D, T, B, idx, mlpMult, blockScales, dropout, prefix)
	if err != nil {
		return wi, err
	}
	prog.Add(x, delta, x)
	layerAgg.apply(prog, x)
	return nextWI, nil
}

func emitMoEParallelDeltaIRWithDropout(prog *Program, spec BlockSpec, xNorm string, wi, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32) (string, int, error) {
	prefix := tmpName(xNorm+"_parallel_moe", idx)
	return emitMoEDeltaIR(prog, spec, xNorm, wi, D, T, B, idx, mlpMult, blockScales, dropout, prefix)
}

func emitMoEDeltaIR(prog *Program, spec BlockSpec, xNorm string, wi, D, T, B, idx int, mlpMult float64, blockScales bool, dropout float32, prefix string) (string, int, error) {
	experts := spec.NumExperts
	topK := effectiveMoETopK(spec)
	expertType, activation, leakySlope, err := moeExpertTypeAndActivation(spec)
	if err != nil {
		return "", wi, err
	}
	ffn := ffnDim(D, mlpMult)
	inputs := []string{xNorm, weightName(wi)}
	wi++
	perExpert, err := moeExpertWeightCount(spec)
	if err != nil {
		return "", wi, err
	}
	for i := 0; i < experts; i++ {
		for j := 0; j < perExpert; j++ {
			inputs = append(inputs, weightName(wi))
			wi++
		}
	}

	delta := prefix + "_delta"
	aux := prefix + "_moe_aux_loss_raw"
	entropy := prefix + "_moe_router_entropy"
	prog.MoEFeedForward(inputs, delta, aux, entropy, B, T, D, experts, topK, expertType, ffn, activation, leakySlope)
	prog.ScalarMul(aux, float32(effectiveMoELoadBalanceLossWeight(spec)), prefix+"_moe_aux_loss")

	if blockScales {
		scaled := prefix + "_scaled"
		prog.Mul(delta, weightName(wi), scaled)
		wi++
		delta = scaled
	}
	if dropout > 0 {
		dropped := prefix + "_dropout"
		prog.Dropout(delta, dropout, dropped)
		delta = dropped
	}
	_ = idx
	return delta, wi, nil
}

func estimateMoEBlockFLOPs(block BlockSpec, B, T, D int, mlpMult float64) int64 {
	if block.NumExperts <= 0 {
		return 0
	}
	ffn := ffnDim(D, mlpMult)
	topK := effectiveMoETopK(block)
	expertType, _, _, err := moeExpertTypeAndActivation(block)
	if err != nil {
		return 0
	}
	total := int64(0)
	total += 2 * i64(B) * i64(T) * i64(D) * i64(block.NumExperts) // router
	switch expertType {
	case moeExpertSwiGLU, moeExpertGEGLU:
		perExpert := 2*2*i64(B)*i64(T)*i64(D)*i64(ffn) + 2*i64(B)*i64(T)*i64(ffn)*i64(D)
		total += i64(topK) * perExpert
	case moeExpertMLP:
		perExpert := 2*i64(B)*i64(T)*i64(D)*i64(ffn) + 2*i64(B)*i64(T)*i64(ffn)*i64(D)
		total += i64(topK) * perExpert
	}
	return total
}
