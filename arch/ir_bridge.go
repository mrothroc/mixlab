package arch

import "fmt"

// BuildIRProgramFromConfig constructs an IR forward-pass program from an
// ArchConfig. The batchSize is derived from config (batch_tokens / seq_len).
func BuildIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	return BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: true,
		HeadUntied:       cfg.MTPUntieEnabled(),
		MTPAuxInactive:   false,
	})
}

// BuildEvalIRProgramFromConfig constructs a next-token-only program for
// validation, inference, and generation. MTP auxiliary losses are intentionally
// excluded because they are training-only signal.
func BuildEvalIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	return buildIRProgramFromConfigWithState(cfg, TrainingProgramState{
		RecurrenceActive: true,
		HeadUntied:       cfg.MTPUntieEnabled(),
	}, nil)
}

// TrainingProgramState selects training-time graph schedules that can switch
// without changing the weight layout.
type TrainingProgramState struct {
	RecurrenceActive bool
	HeadUntied       bool
	MTPAuxInactive   bool
}

// BuildTrainingIRProgramFromConfig constructs a training program with MTP
// auxiliary losses enabled when configured unless state disables them.
func BuildTrainingIRProgramFromConfig(cfg *ArchConfig, state TrainingProgramState) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	return buildIRProgramFromConfigWithState(cfg, state, cfg.MTP)
}

// BuildPreActivationIRProgramFromConfig constructs the recurrence-inactive
// training program for configs that delay recurrence activation. The program
// keeps the full recurrence weight layout, but emits only the first occurrence
// of each recurrence root in order.
func BuildPreActivationIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	return BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
		RecurrenceActive: false,
		HeadUntied:       cfg.MTPUntieEnabled(),
		MTPAuxInactive:   false,
	})
}

func buildIRProgramFromConfigWithState(cfg *ArchConfig, state TrainingProgramState, mtp *MTPSpec) (*Program, error) {
	batchSize := 1
	if cfg.Training.BatchTokens > 0 && cfg.SeqLen > 0 {
		batchSize = cfg.Training.BatchTokens / cfg.SeqLen
		if batchSize <= 0 {
			batchSize = 1
		}
	}

	var executionOrder []int
	if !state.RecurrenceActive {
		order, err := uniqueRecurrenceExecutionOrder(cfg.Blocks, cfg.Recurrence)
		if err != nil {
			return nil, err
		}
		executionOrder = order
	}
	reserveHead := cfg.ReservesUntiedHeadWeight()
	useTiedHead := cfg.TieEmbeddings && !state.HeadUntied
	activeMTP := mtp
	if state.MTPAuxInactive {
		activeMTP = nil
	}

	return buildIRProgramWithDropoutNgramsOrderAndSmear(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		batchSize,
		cfg.EffectiveMLPMult(),
		cfg.TieEmbeddings,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
		cfg.LogitSoftcap,
		cfg.Dropout,
		cfg.Blocks,
		cfg.Recurrence,
		executionOrder,
		activeMTP,
		reserveHead,
		useTiedHead,
		cfg.smearEmbeddingOptions(),
		cfg.Backout,
	)
}

// CountIRWeightsFromConfig returns the expected number of IR weight tensors.
func CountIRWeightsFromConfig(cfg *ArchConfig) (int, error) {
	if cfg == nil {
		return 0, fmt.Errorf("nil config")
	}

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		return 0, err
	}
	return len(metas), nil
}

func convertBlockSpecs(specs []BlockSpec) []BlockSpec {
	if len(specs) == 0 {
		return nil
	}
	out := make([]BlockSpec, len(specs))
	copy(out, specs)
	return out
}
