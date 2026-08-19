package arch

import "fmt"

// Weight shapes for the discrete multi-codebook input adapter.

func collectDiscreteCodebookWeightShapesFromConfig(cfg *ArchConfig) ([]WeightMeta, error) {
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	return collectDiscreteCodebookWeightShapesWithRefs(cfg, refs)
}

func collectDiscreteCodebookWeightShapesWithRefs(cfg *ArchConfig, refs []int) ([]WeightMeta, error) {
	rows := cfg.CodebookEmbeddingRows()
	metas, err := collectWeightShapesWithRefsHeadLayoutFeaturesNorm(
		cfg.ModelDim,
		rows,
		cfg.SeqLen,
		cfg.EffectiveMLPMult(),
		false,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.EffectivePositionalEmbedding(),
		cfg.EffectiveMaxPositions(),
		0, 0, 0, 0, 0, 0,
		cfg.Blocks,
		refs,
		cfg.EffectiveNormSpec(),
		cfg.EffectiveNormPlacement(),
		cfg.FFNInternalNorm,
		cfg.EffectiveFinalNorm(),
	)
	if err != nil {
		return nil, err
	}
	metas[0].Name = "input_adapter_codebook_embedding"
	metas[0].InitMode = "torch_embedding_normal_1"

	fixed := fixedWeightCountWithHeadAndNorm(false, cfg.EffectiveNormSpec(), cfg.EffectiveFinalNorm())
	extra := discreteCodebookExtraWeightShapes(cfg)
	out := make([]WeightMeta, 0, len(metas)+len(extra)+len(backoutWeightShapes(cfg.Backout))+len(classificationWeightShapes(cfg.ModelDim, cfg.Training.Classification)))
	out = append(out, metas[:fixed]...)
	out = append(out, extra...)
	out = append(out, metas[fixed:]...)
	out = append(out, backoutWeightShapes(cfg.Backout)...)
	layerAggregationMetas, err := layerAggregationWeightShapes(cfg.Blocks, cfg.EffectiveLayerAggregation())
	if err != nil {
		return nil, err
	}
	out = append(out, layerAggregationMetas...)
	out = append(out, classificationWeightShapes(cfg.ModelDim, cfg.Training.Classification)...)
	return out, nil
}
