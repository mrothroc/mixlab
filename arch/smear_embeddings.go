package arch

import (
	"fmt"
	"strings"
)

const (
	SmearEmbeddingsGatePR130                 = "pr130"
	SmearEmbeddingsGatePerChannel            = "per_channel"
	SmearEmbeddingsGatePerPositionPerChannel = "per_position_per_channel"

	smearPR130GateInputDim = 12
)

type smearEmbeddingOptions struct {
	Enabled   bool
	GateShape string
}

func disabledSmearEmbeddingOptions() smearEmbeddingOptions {
	return smearEmbeddingOptions{}
}

func (c *ArchConfig) smearEmbeddingOptions() smearEmbeddingOptions {
	if c == nil || !c.SmearEmbeddings {
		return disabledSmearEmbeddingOptions()
	}
	return smearEmbeddingOptions{
		Enabled:   true,
		GateShape: c.EffectiveSmearEmbeddingsGateShape(),
	}
}

// EffectiveSmearEmbeddingsGateShape returns the configured smear gate variant.
// The default is the modded-nanogpt PR #130 dynamic gate over the first 12
// embedding channels, because that is the upstream record implementation.
func (c *ArchConfig) EffectiveSmearEmbeddingsGateShape() string {
	if c == nil || !c.SmearEmbeddings {
		return ""
	}
	gateShape := strings.ToLower(strings.TrimSpace(c.SmearEmbeddingsGateShape))
	if gateShape == "" {
		return SmearEmbeddingsGatePR130
	}
	return gateShape
}

func validateSmearEmbeddings(cfg *ArchConfig, source string) error {
	if cfg == nil || !cfg.SmearEmbeddings {
		if cfg != nil {
			cfg.SmearEmbeddingsGateShape = ""
		}
		return nil
	}
	if cfg.SeqLen <= 1 {
		return fmt.Errorf("config %q enables smear_embeddings but seq_len=%d (must be > 1)", source, cfg.SeqLen)
	}
	gateShape := cfg.EffectiveSmearEmbeddingsGateShape()
	switch gateShape {
	case SmearEmbeddingsGatePR130:
		if cfg.ModelDim < smearPR130GateInputDim {
			return fmt.Errorf("config %q enables smear_embeddings_gate_shape=%q but model_dim=%d (must be >= %d)", source, gateShape, cfg.ModelDim, smearPR130GateInputDim)
		}
	case SmearEmbeddingsGatePerChannel, SmearEmbeddingsGatePerPositionPerChannel:
	default:
		return fmt.Errorf("config %q has invalid smear_embeddings_gate_shape=%q (must be %q, %q, or %q)", source, cfg.SmearEmbeddingsGateShape, SmearEmbeddingsGatePR130, SmearEmbeddingsGatePerChannel, SmearEmbeddingsGatePerPositionPerChannel)
	}
	cfg.SmearEmbeddingsGateShape = gateShape
	return nil
}

func smearEmbeddingWeightShapes(D, T int, opts smearEmbeddingOptions) ([]WeightMeta, error) {
	if !opts.Enabled {
		return nil, nil
	}
	switch opts.GateShape {
	case SmearEmbeddingsGatePR130:
		if D < smearPR130GateInputDim {
			return nil, fmt.Errorf("smear_embeddings_gate_shape=%q requires model_dim >= %d", opts.GateShape, smearPR130GateInputDim)
		}
		return []WeightMeta{
			{Name: "smear_gate", Shape: []int{smearPR130GateInputDim, 1}, InitZero: true},
			{Name: "smear_scale", Shape: []int{1}, InitZero: true},
		}, nil
	case SmearEmbeddingsGatePerChannel:
		return []WeightMeta{{Name: "smear_gate", Shape: []int{D}, InitZero: true}}, nil
	case SmearEmbeddingsGatePerPositionPerChannel:
		return []WeightMeta{{Name: "smear_gate", Shape: []int{T, D}, InitZero: true}}, nil
	default:
		return nil, fmt.Errorf("invalid smear_embeddings_gate_shape=%q", opts.GateShape)
	}
}

func emitSmearEmbeddingIR(prog *Program, input string, T, D, wi int, opts smearEmbeddingOptions) (string, int, error) {
	if !opts.Enabled {
		return input, wi, nil
	}
	if T <= 1 {
		return "", wi, fmt.Errorf("smear_embeddings requires seq_len > 1")
	}

	first := input + "_smear_first"
	tail := input + "_smear_tail"
	prev := input + "_smear_prev"
	smear := input + "_smear_term"
	tailOut := input + "_smear_tail_out"
	output := input + "_smeared"

	prog.Slice(input, 0, 1, 1, 1, first)
	prog.Slice(input, 1, T, 1, 1, tail)
	prog.Slice(input, 0, T-1, 1, 1, prev)

	switch opts.GateShape {
	case SmearEmbeddingsGatePR130:
		if D < smearPR130GateInputDim {
			return "", wi, fmt.Errorf("smear_embeddings_gate_shape=%q requires model_dim >= %d", opts.GateShape, smearPR130GateInputDim)
		}
		gateInput := input + "_smear_gate_input"
		gateLogits := input + "_smear_gate_logits"
		gateSigmoid := input + "_smear_gate_sigmoid"
		gate := input + "_smear_gate"
		prog.Slice(tail, 0, smearPR130GateInputDim, 1, 2, gateInput)
		prog.MatMul(gateInput, weightName(wi), gateLogits)
		wi++
		prog.Sigmoid(gateLogits, gateSigmoid)
		prog.Mul(gateSigmoid, weightName(wi), gate)
		wi++
		prog.Mul(gate, prev, smear)
	case SmearEmbeddingsGatePerChannel:
		prog.Mul(prev, weightName(wi), smear)
		wi++
	case SmearEmbeddingsGatePerPositionPerChannel:
		gateTail := input + "_smear_gate_tail"
		prog.Slice(weightName(wi), 1, T, 1, 0, gateTail)
		wi++
		prog.Mul(prev, gateTail, smear)
	default:
		return "", wi, fmt.Errorf("invalid smear_embeddings_gate_shape=%q", opts.GateShape)
	}

	prog.Add(tail, smear, tailOut)
	prog.Concat(first, tailOut, 1, output)
	return output, wi, nil
}
