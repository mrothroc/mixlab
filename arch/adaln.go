package arch

import "fmt"

type adaLNBuildState struct {
	weightStart int
	condDim     int
	modelDim    int
}

func newAdaLNBuildState(enabled bool, weightStart, condDim, modelDim int) *adaLNBuildState {
	if !enabled {
		return nil
	}
	return &adaLNBuildState{weightStart: weightStart, condDim: condDim, modelDim: modelDim}
}

func (a *adaLNBuildState) weightsPerBlock() int {
	if a == nil {
		return 0
	}
	return 2
}

func (a *adaLNBuildState) apply(prog *Program, x string, B, T, blockIdx int, label string) string {
	if a == nil {
		return x
	}
	prefix := fmt.Sprintf("%s_adaln_%d_%s", x, blockIdx, label)
	w1 := weightName(a.weightStart + blockIdx*a.weightsPerBlock())
	w2 := weightName(a.weightStart + blockIdx*a.weightsPerBlock() + 1)
	timestep2 := prefix + "_timestep2"
	hidden := prefix + "_hidden"
	hiddenAct := prefix + "_hidden_act"
	mod := prefix + "_mod"
	scale := prefix + "_scale"
	shift := prefix + "_shift"
	one := prefix + "_one"
	scalePlusOne := prefix + "_scale_plus_one"
	scale3 := prefix + "_scale3"
	shift3 := prefix + "_shift3"
	x3 := prefix + "_x3"
	scaled := prefix + "_scaled"
	shifted := prefix + "_shifted"
	out := prefix + "_out"

	prog.Reshape("diffusion_timestep", []int{B, 1}, timestep2)
	prog.MatMul(timestep2, w1, hidden)
	prog.GELU(hidden, hiddenAct)
	prog.MatMul(hiddenAct, w2, mod)
	prog.Slice(mod, 0, a.modelDim, 1, 1, scale)
	prog.Slice(mod, a.modelDim, 2*a.modelDim, 1, 1, shift)
	prog.Full([]int{1}, 1.0, one)
	prog.Add(scale, one, scalePlusOne)
	prog.Reshape(x, []int{B, T, a.modelDim}, x3)
	prog.Reshape(scalePlusOne, []int{B, 1, a.modelDim}, scale3)
	prog.Reshape(shift, []int{B, 1, a.modelDim}, shift3)
	prog.Mul(x3, scale3, scaled)
	prog.Add(scaled, shift3, shifted)
	prog.Reshape(shifted, []int{B * T, a.modelDim}, out)
	return out
}
