package arch

import "fmt"

// BuildIRProgramFromConfig constructs an IR forward-pass program from an
// ArchConfig. The batchSize is derived from config (batch_tokens / seq_len).
func BuildIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	return buildIRProgramFromConfigWithOrder(cfg, nil)
}

// BuildPreActivationIRProgramFromConfig constructs the recurrence-inactive
// training program for configs that delay recurrence activation. The program
// keeps the full recurrence weight layout, but emits only the first occurrence
// of each recurrence root in order.
func BuildPreActivationIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	order, err := uniqueRecurrenceExecutionOrder(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return nil, err
	}
	if len(order) == 0 {
		return BuildIRProgramFromConfig(cfg)
	}
	return buildIRProgramFromConfigWithOrder(cfg, order)
}

func buildIRProgramFromConfigWithOrder(cfg *ArchConfig, executionOrder []int) (*Program, error) {
	batchSize := 1
	if cfg.Training.BatchTokens > 0 && cfg.SeqLen > 0 {
		batchSize = cfg.Training.BatchTokens / cfg.SeqLen
		if batchSize <= 0 {
			batchSize = 1
		}
	}

	return buildIRProgramWithDropoutNgramsAndOrder(
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
	)
}

// CountIRWeightsFromConfig returns the expected number of IR weight tensors.
func CountIRWeightsFromConfig(cfg *ArchConfig) (int, error) {
	if cfg == nil {
		return 0, fmt.Errorf("nil config")
	}

	return countWeightsWithNgramsRecurrenceAndParallel(
		cfg.ModelDim,
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
		cfg.Blocks,
		cfg.Recurrence,
	)
}

func convertBlockSpecs(specs []BlockSpec) []BlockSpec {
	if len(specs) == 0 {
		return nil
	}
	out := make([]BlockSpec, len(specs))
	copy(out, specs)
	return out
}
