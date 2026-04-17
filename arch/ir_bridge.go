package arch

import "fmt"

// BuildIRProgramFromConfig constructs an IR forward-pass program from an
// ArchConfig. The batchSize is derived from config (batch_tokens / seq_len).
func BuildIRProgramFromConfig(cfg *ArchConfig) (*Program, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	batchSize := 1
	if cfg.Training.BatchTokens > 0 && cfg.SeqLen > 0 {
		batchSize = cfg.Training.BatchTokens / cfg.SeqLen
		if batchSize <= 0 {
			batchSize = 1
		}
	}

	return BuildIRProgramWithBigramRecurrenceAndParallel(
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
		cfg.LogitSoftcap,
		cfg.Blocks,
		cfg.Recurrence,
	)
}

// CountIRWeightsFromConfig returns the expected number of IR weight tensors.
func CountIRWeightsFromConfig(cfg *ArchConfig) (int, error) {
	if cfg == nil {
		return 0, fmt.Errorf("nil config")
	}

	return CountWeightsWithBigramRecurrenceAndParallel(
		cfg.ModelDim,
		cfg.EffectiveMLPMult(),
		cfg.TieEmbeddings,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
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
