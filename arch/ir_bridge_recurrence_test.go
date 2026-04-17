package arch

import "testing"

func TestBuildIRProgramFromConfig_PassesRecurrence(t *testing.T) {
	cfg := &ArchConfig{
		Name:       "bridge_recurrence",
		ModelDim:   64,
		VocabSize:  256,
		SeqLen:     32,
		Blocks:     []BlockSpec{{Type: "plain", Heads: 4}, {Type: "swiglu"}, {Type: "plain", Heads: 4}, {Type: "swiglu"}},
		Recurrence: []int{0, 1, 0, 1},
		Training:   TrainingSpec{BatchTokens: 32},
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != 14 {
		t.Fatalf("NumWeights=%d want 14", prog.NumWeights)
	}

	shapes, err := CollectWeightShapesWithBigramRecurrenceAndParallel(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
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
	if err != nil {
		t.Fatalf("CollectWeightShapesWithBigramRecurrenceAndParallel: %v", err)
	}
	if len(shapes) != 14 {
		t.Fatalf("weight shapes len=%d want 14", len(shapes))
	}
}
