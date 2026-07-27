package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	S4DInitLin = "s4d-lin"

	S4DOutputTransformNone = "none"
	S4DOutputTransformGLU  = "glu"

	defaultS4DStateSize = 64
	defaultS4DDTMin     = 0.001
	defaultS4DDTMax     = 0.1
)

func effectiveS4DOutputTransform(spec BlockSpec) string {
	switch strings.ToLower(strings.TrimSpace(spec.OutputTransform)) {
	case "", S4DOutputTransformNone:
		return S4DOutputTransformNone
	case S4DOutputTransformGLU:
		return S4DOutputTransformGLU
	default:
		return strings.ToLower(strings.TrimSpace(spec.OutputTransform))
	}
}

func effectiveS4DStateSize(spec BlockSpec) int {
	if spec.StateSize != 0 {
		return spec.StateSize
	}
	return defaultS4DStateSize
}

func effectiveS4DInit(spec BlockSpec) string {
	switch strings.ToLower(strings.TrimSpace(spec.Init)) {
	case "", S4DInitLin:
		return S4DInitLin
	default:
		return strings.ToLower(strings.TrimSpace(spec.Init))
	}
}

func effectiveS4DDTRange(spec BlockSpec) (float64, float64) {
	dtMin := spec.DTMin
	if dtMin == 0 {
		dtMin = defaultS4DDTMin
	}
	dtMax := spec.DTMax
	if dtMax == 0 {
		dtMax = defaultS4DDTMax
	}
	return dtMin, dtMax
}

func s4dWeightShapesWithOptions(spec BlockSpec, D int, opts EmitOptions) ([]WeightMeta, error) {
	stateSize := effectiveS4DStateSize(spec)
	if stateSize <= 0 || stateSize%2 != 0 {
		return nil, fmt.Errorf("s4d state_size must be positive and even, got %d", stateSize)
	}
	if effectiveS4DInit(spec) != S4DInitLin {
		return nil, fmt.Errorf("s4d init %q is unsupported", spec.Init)
	}
	dtMin, dtMax := effectiveS4DDTRange(spec)
	if !(dtMin > 0 && dtMax > dtMin) {
		return nil, fmt.Errorf("s4d requires 0 < dt_min < dt_max")
	}
	norm := normSpecOrDefault(opts.Norm)
	placement := normPlacementOrDefault(opts.NormPlacement)
	statePairs := stateSize / 2
	metas := make([]WeightMeta, 0, 10)
	if placement == NormPlacementPre || placement == NormPlacementSandwich {
		metas = append(metas, normWeights("s4d_norm", D, norm)...)
	}
	metas = append(metas,
		WeightMeta{Name: "s4d_log_dt", Shape: []int{D}, InitMode: "s4d_log_dt", DtMin: dtMin, DtMax: dtMax},
		WeightMeta{Name: "s4d_log_A_real", Shape: []int{D, statePairs}, InitValue: float32(math.Log(0.5))},
		WeightMeta{Name: "s4d_A_imag", Shape: []int{D, statePairs}, InitMode: "s4d_A_imag_lin"},
		WeightMeta{Name: "s4d_C_real", Shape: []int{D, statePairs}, InitMode: "s4d_C_normal"},
		WeightMeta{Name: "s4d_C_imag", Shape: []int{D, statePairs}, InitMode: "s4d_C_normal"},
		WeightMeta{Name: "s4d_D", Shape: []int{D}, InitMode: "s4d_D_normal"},
	)
	if effectiveS4DOutputTransform(spec) == S4DOutputTransformGLU {
		metas = append(metas,
			WeightMeta{Name: "s4d_out_proj", Shape: []int{D, 2 * D}},
			WeightMeta{Name: "s4d_out_bias", Shape: []int{2 * D}, InitZero: true},
		)
	}
	if placement == NormPlacementPost || placement == NormPlacementSandwich {
		metas = append(metas, normWeights("s4d_post_norm", D, norm)...)
	}
	if opts.BlockScales {
		metas = append(metas, residualScaleWeightMeta(spec, "s4d_scale", D))
	}
	return metas, nil
}

func s4dWeightShapes(spec BlockSpec, D, _, _, _ int) ([]WeightMeta, error) {
	return s4dWeightShapesWithOptions(spec, D, EmitOptions{
		Norm:          defaultNormSpec(),
		NormPlacement: NormPlacementPre,
	})
}

func s4dWeightCount(spec BlockSpec, blockScales, _ bool) (int, error) {
	metas, err := s4dWeightShapesWithOptions(spec, 1, EmitOptions{
		Norm:          defaultNormSpec(),
		NormPlacement: NormPlacementPre,
		BlockScales:   blockScales,
	})
	return len(metas), err
}

func emitS4DIR(
	prog *Program,
	spec BlockSpec,
	stream string,
	wi, D, T, B, _ int,
	idx int,
	opts EmitOptions,
) (int, error) {
	stateSize := effectiveS4DStateSize(spec)
	norm := normSpecOrDefault(opts.Norm)
	placement := normPlacementOrDefault(opts.NormPlacement)
	prefix := tmpName(stream+"_s4d", idx)
	xNorm := prefix + "_norm"
	s4dOut := prefix + "_out"
	kernel := prefix + "_kernel"
	activated := prefix + "_gelu"
	postNorm := prefix + "_post_norm"
	scaled := prefix + "_scaled"
	dropped := prefix + "_dropout"
	innerDropped := prefix + "_inner_dropout"
	projected := prefix + "_projected"
	projectedBiased := prefix + "_projected_biased"
	gluValue := prefix + "_glu_value"
	gluGate := prefix + "_glu_gate"
	gluGateSigmoid := prefix + "_glu_gate_sigmoid"
	gluOut := prefix + "_glu_out"

	input := stream
	if placement == NormPlacementPre || placement == NormPlacementSandwich {
		var err error
		wi, err = emitNamedNormIR(prog, stream, wi, xNorm, norm)
		if err != nil {
			return wi, err
		}
		input = xNorm
	}

	prog.S4D(
		input,
		weightName(wi),
		weightName(wi+1),
		weightName(wi+2),
		weightName(wi+3),
		weightName(wi+4),
		weightName(wi+5),
		s4dOut,
		kernel,
		B,
		T,
		D,
		stateSize,
		0,
	)
	wi += 6
	if effectiveS4DOutputTransform(spec) == S4DOutputTransformGLU {
		prog.GELUExact(s4dOut, activated)
	} else {
		prog.GELU(s4dOut, activated)
	}
	delta := activated
	if effectiveS4DOutputTransform(spec) == S4DOutputTransformGLU {
		if opts.Dropout > 0 {
			prog.Dropout(delta, opts.Dropout, innerDropped)
			delta = innerDropped
		}
		prog.MatMul(delta, weightName(wi), projected)
		prog.Add(projected, weightName(wi+1), projectedBiased)
		wi += 2
		prog.Slice(projectedBiased, 0, D, 1, 1, gluValue)
		prog.Slice(projectedBiased, D, 2*D, 1, 1, gluGate)
		prog.Sigmoid(gluGate, gluGateSigmoid)
		prog.Mul(gluValue, gluGateSigmoid, gluOut)
		delta = gluOut
	}

	if placement == NormPlacementPost || placement == NormPlacementSandwich {
		var err error
		wi, err = emitNamedNormIR(prog, delta, wi, postNorm, norm)
		if err != nil {
			return wi, err
		}
		delta = postNorm
	}
	if opts.BlockScales {
		prog.Mul(delta, weightName(wi), scaled)
		wi++
		delta = scaled
	}
	if opts.Dropout > 0 {
		prog.Dropout(delta, opts.Dropout, dropped)
		delta = dropped
	}
	prog.Add(stream, delta, stream)
	return wi, nil
}

func init() {
	RegisterBlock("s4d", blockRegistration{
		Emitter:      emitS4DIR,
		WeightCount:  s4dWeightCount,
		WeightShapes: s4dWeightShapes,
		weightShapesWithOptions: func(spec BlockSpec, D, _, _, _ int, opts EmitOptions) ([]WeightMeta, error) {
			return s4dWeightShapesWithOptions(spec, D, opts)
		},
	})
}
