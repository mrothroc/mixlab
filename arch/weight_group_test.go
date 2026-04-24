package arch

import "testing"

func testWeightGroupConfig(shared bool) *ArchConfig {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
		{Type: "plain", Heads: 4},
		{Type: "swiglu"},
	}
	if shared {
		blocks[0].WeightGroup = "attn"
		blocks[1].WeightGroup = "mlp"
		blocks[2].WeightGroup = "attn"
		blocks[3].WeightGroup = "mlp"
	}
	return &ArchConfig{
		Name:      "weight-group",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		Blocks:    blocks,
		Training:  TrainingSpec{BatchTokens: 32},
	}
}

func TestWeightSharingIR(t *testing.T) {
	cfg := testWeightGroupConfig(true)

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != 14 {
		t.Fatalf("NumWeights=%d want 14", prog.NumWeights)
	}

	plain0 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_0_x_norm")
	plain1 := weightInputForOutput(t, prog, OpRMSNorm, "x_attn_2_x_norm")
	if plain0 != "w3" || plain1 != plain0 {
		t.Fatalf("plain weights %q and %q want both w3", plain0, plain1)
	}

	swiglu0 := weightInputForOutput(t, prog, OpRMSNorm, "x_swiglu_1_x_norm")
	swiglu1 := weightInputForOutput(t, prog, OpRMSNorm, "x_swiglu_3_x_norm")
	if swiglu0 != "w10" || swiglu1 != swiglu0 {
		t.Fatalf("swiglu weights %q and %q want both w10", swiglu0, swiglu1)
	}
}

func TestWeightSharingNoGroup(t *testing.T) {
	cfg := testWeightGroupConfig(false)

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if prog.NumWeights != 25 {
		t.Fatalf("NumWeights=%d want 25", prog.NumWeights)
	}
}

func TestCollectWeightShapesWithWeightGroup(t *testing.T) {
	cfg := testWeightGroupConfig(true)

	metas, err := CollectWeightShapesWithBigramRecurrenceAndParallel(
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
	if len(metas) != 14 {
		t.Fatalf("len(metas)=%d want 14", len(metas))
	}
}

func TestWeightSharingFLOPs(t *testing.T) {
	sharedCfg := testWeightGroupConfig(true)
	unsharedCfg := testWeightGroupConfig(false)
	shared := EstimateFLOPs(sharedCfg)
	unshared := EstimateFLOPs(unsharedCfg)

	if shared.ForwardFLOPs != unshared.ForwardFLOPs {
		t.Fatalf("shared ForwardFLOPs=%d want %d", shared.ForwardFLOPs, unshared.ForwardFLOPs)
	}
	fixed, err := collectWeightShapesWithRefs(
		sharedCfg.ModelDim,
		sharedCfg.VocabSize,
		sharedCfg.SeqLen,
		sharedCfg.EffectiveMLPMult(),
		sharedCfg.TieEmbeddings,
		sharedCfg.BlockScales,
		sharedCfg.ResidMix,
		sharedCfg.UNet,
		sharedCfg.ParallelResidual,
		sharedCfg.BigramVocabSize,
		sharedCfg.EffectiveBigramDim(),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("collectWeightShapesWithRefs: %v", err)
	}
	fixedParams := countWeightMetaElements(fixed)
	sharedBlockParams := shared.ParamCount - fixedParams
	unsharedBlockParams := unshared.ParamCount - fixedParams
	if sharedBlockParams*2 != unsharedBlockParams {
		t.Fatalf("shared block params=%d want half of unshared block params=%d", sharedBlockParams, unsharedBlockParams)
	}
	if shared.ExpandedParamCount != unshared.ParamCount {
		t.Fatalf("shared ExpandedParamCount=%d want %d", shared.ExpandedParamCount, unshared.ParamCount)
	}
}
