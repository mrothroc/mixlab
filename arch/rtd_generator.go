package arch

import "fmt"

const (
	RTDGeneratorPrefix     = "rtd_generator"
	RTDGeneratorLogitsName = "rtd_generator_logits"
)

func rtdDedicatedGeneratorSpec(cfg *ArchConfig) *RTDDedicatedGenerator {
	if cfg == nil || cfg.Training.RTD == nil || !cfg.Training.RTD.DedicatedGeneratorEnabled() {
		return nil
	}
	return cfg.Training.RTD.DedicatedGenerator
}

func rtdDedicatedGeneratorBlocks(spec *RTDDedicatedGenerator) []BlockSpec {
	if spec == nil || spec.Layers <= 0 {
		return nil
	}
	blocks := make([]BlockSpec, spec.Layers)
	for i := range blocks {
		blocks[i] = BlockSpec{
			Type:          "plain",
			Heads:         spec.Heads,
			AttentionMask: AttentionMaskBidirectional,
			FFNActivation: PlainFFNActivationGEGLU,
		}
	}
	return blocks
}

func rtdDedicatedGeneratorWeightShapes(cfg *ArchConfig) ([]WeightMeta, error) {
	spec := rtdDedicatedGeneratorSpec(cfg)
	if spec == nil {
		return nil, nil
	}
	D := spec.ModelDim
	T := cfg.SeqLen
	V := cfg.VocabSize
	B := 1
	norm := cfg.EffectiveNormSpec()
	opts := EmitOptions{
		MLPMult:             spec.MLPMult,
		Norm:                norm,
		NormPlacement:       NormPlacementPre,
		PositionalEmbedding: PositionalEmbeddingRope,
	}
	shapes := []WeightMeta{{Name: RTDGeneratorPrefix + "_embed", Shape: []int{V, D}}}
	for layerIdx, block := range rtdDedicatedGeneratorBlocks(spec) {
		blockShapes, err := blockWeightShapesWithEmitOptions(block, D, T, B, V, opts)
		if err != nil {
			return nil, fmt.Errorf("rtd generator layer %d: %w", layerIdx, err)
		}
		shapes = append(shapes, prefixedWeightMetas(fmt.Sprintf("%s_l%d_", RTDGeneratorPrefix, layerIdx), blockShapes)...)
	}
	shapes = append(shapes, prefixedWeightMetas(RTDGeneratorPrefix+"_", normWeights("final_norm", D, norm))...)
	shapes = append(shapes,
		WeightMeta{Name: RTDGeneratorPrefix + "_mlm_dense", Shape: []int{D, D}},
		WeightMeta{Name: RTDGeneratorPrefix + "_mlm_dense_bias", Shape: []int{D}, InitZero: true},
		WeightMeta{Name: RTDGeneratorPrefix + "_mlm_output_bias", Shape: []int{V}, InitZero: true},
	)
	return shapes, nil
}

func prefixedWeightMetas(prefix string, metas []WeightMeta) []WeightMeta {
	out := make([]WeightMeta, len(metas))
	for i, meta := range metas {
		out[i] = meta
		out[i].Name = prefix + meta.Name
	}
	return out
}

func emitRTDDedicatedGeneratorIR(prog *Program, cfg *ArchConfig, wi, rawBatch int, dropout float32) (string, string, int, error) {
	spec := rtdDedicatedGeneratorSpec(cfg)
	if spec == nil {
		return "", "", wi, nil
	}
	if prog == nil {
		return "", "", wi, fmt.Errorf("nil program")
	}
	T := cfg.SeqLen
	D := spec.ModelDim
	V := cfg.VocabSize
	rawRows := rawBatch * T
	prog.DeclareInput("rtd_generator_tokens", TensorInt32, []int{rawBatch, T})
	prog.DeclareInput("rtd_generator_targets", TensorInt32, []int{rawRows})
	prog.DeclareInput("rtd_generator_loss_mask", TensorFloat32, []int{rawRows})
	embedIdx := wi
	prog.Embed(weightName(wi), "rtd_generator_tokens", "rtd_generator_embed_out")
	wi++
	prog.Reshape("rtd_generator_embed_out", []int{rawRows, D}, "rtd_generator_x")
	stream := "rtd_generator_x"
	opts := EmitOptions{
		MLPMult:             spec.MLPMult,
		Dropout:             dropout,
		AttnDropout:         dropout,
		Norm:                cfg.EffectiveNormSpec(),
		NormPlacement:       NormPlacementPre,
		PositionalEmbedding: PositionalEmbeddingRope,
		KVCache:             make(map[int]BlockKVOutputs),
	}
	for layerIdx, block := range rtdDedicatedGeneratorBlocks(spec) {
		var err error
		wi, err = EmitBlock(prog, block, stream, wi, D, T, rawBatch, V, layerIdx, opts)
		if err != nil {
			return "", "", wi, fmt.Errorf("rtd generator layer %d: %w", layerIdx, err)
		}
		stream = "x"
	}
	wi, err := emitNamedNormIR(prog, stream, wi, "rtd_generator_final_norm", cfg.EffectiveNormSpec())
	if err != nil {
		return "", "", wi, err
	}
	_, wi, err = emitBERTMLMHeadIRTiedTo(prog, RTDGeneratorPrefix+"_mlm", "rtd_generator_final_norm", wi, weightName(embedIdx), D, V, cfg.EffectiveNormSpec().Eps, dropout)
	if err != nil {
		return "", "", wi, err
	}
	prog.ScalarMul(RTDGeneratorPrefix+"_mlm_logits", 1.0, RTDGeneratorLogitsName)
	return RTDGeneratorLogitsName, "rtd_generator_loss_mask", wi, nil
}
