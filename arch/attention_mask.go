package arch

import "fmt"

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func ensureProgramInput(prog *Program, name string, dtype int, shape []int) error {
	for _, in := range prog.Inputs {
		if in.Name != name {
			continue
		}
		if in.DType != dtype || !sameIntShape(in.Shape, shape) {
			return fmt.Errorf("input %q already declared with dtype=%d shape=%v, want dtype=%d shape=%v", name, in.DType, in.Shape, dtype, shape)
		}
		return nil
	}
	prog.DeclareInput(name, dtype, shape)
	return nil
}

func emitPlainAttentionMaskIR(prog *Program, scores, output, attentionMask string, B, T, windowSize int, segmentMask bool) (string, error) {
	if attentionMask == AttentionMaskBlockDiffusion {
		if segmentMask {
			return "", fmt.Errorf("block_diffusion attention mask cannot be combined with segment attention mask")
		}
		if windowSize > 0 {
			return "", fmt.Errorf("block_diffusion attention mask cannot be combined with window_size")
		}
		if err := ensureProgramInput(prog, "diffusion_block_start", TensorInt32, []int{B}); err != nil {
			return "", err
		}
		if err := ensureProgramInput(prog, "diffusion_block_end", TensorInt32, []int{B}); err != nil {
			return "", err
		}
		prog.BlockDiffusionMask(scores, "diffusion_block_start", "diffusion_block_end", T, output)
		return output, nil
	}
	if segmentMask {
		switch attentionMask {
		case AttentionMaskCausal:
			prog.SegmentAttentionMask(scores, "segment_ids", "", T, windowSize, SegmentMaskModeCausal, output)
		case AttentionMaskHybridExample:
			prog.SegmentAttentionMask(scores, "segment_ids", "attention_causal_mask", T, windowSize, SegmentMaskModeSelectiveCausal, output)
		case AttentionMaskBidirectional, AttentionMaskNone:
			prog.SegmentAttentionMask(scores, "segment_ids", "", T, windowSize, SegmentMaskModeNone, output)
		default:
			return "", fmt.Errorf("invalid attention_mask=%q", attentionMask)
		}
		return output, nil
	}
	switch attentionMask {
	case AttentionMaskCausal:
		prog.CausalMask(scores, T, windowSize, output)
		return output, nil
	case AttentionMaskHybridExample:
		prog.SelectiveCausalMask(scores, "attention_causal_mask", T, windowSize, output)
		return output, nil
	case AttentionMaskBidirectional, AttentionMaskNone:
		return scores, nil
	default:
		return "", fmt.Errorf("invalid attention_mask=%q", attentionMask)
	}
}
