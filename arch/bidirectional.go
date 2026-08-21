package arch

import "fmt"

const sequenceValidMaskInput = "sequence_valid_mask"

func supportsBidirectionalMixer(spec BlockSpec) bool {
	switch blockTypeKey(spec) {
	case "s4d", "mamba3-canonical", "gated_deltanet":
		return true
	default:
		return false
	}
}

func containsBidirectionalMixer(blocks []BlockSpec) bool {
	for _, block := range blocks {
		if block.Bidirectional {
			return true
		}
	}
	return false
}

func programDeclaresInput(prog *Program, name string) bool {
	for _, input := range prog.Inputs {
		if input.Name == name {
			return true
		}
	}
	return false
}

func bidirectionalValidMask(prog *Program, B, T int, prefix string) string {
	if programDeclaresInput(prog, sequenceValidMaskInput) {
		return sequenceValidMaskInput
	}
	mask := prefix + "_all_valid"
	prog.Full([]int{B, T}, 1, mask)
	return mask
}

func maskSequenceValidIR(prog *Program, input, validMask, output string, B, T int) {
	flatMask := output + "_mask"
	prog.Reshape(validMask, []int{B * T, 1}, flatMask)
	prog.Mul(input, flatMask, output)
}

func validateBidirectionalMixers(cfg *ArchConfig, source string) error {
	if cfg == nil || !containsBidirectionalMixer(cfg.Blocks) {
		return nil
	}
	switch objective := cfg.Training.EffectiveObjective(); objective {
	case ObjectiveClassification, ObjectiveMLM, ObjectiveMNTP:
		// These objectives permit future-token context.
	default:
		return fmt.Errorf("config %q uses bidirectional recurrent mixing with training.objective=%q; bidirectional recurrent mixers are supported only for classification, mlm, or mntp because they expose future-token context", source, objective)
	}
	if cfg.Training.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("config %q combines bidirectional recurrent mixing with packed segment attention; recurrent state does not reset at segment boundaries in v1", source)
	}
	return nil
}
