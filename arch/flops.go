package arch

import (
	"fmt"
	"strings"
)

// FLOPsEstimate summarizes analytical floating-point operation counts for an
// architecture and training batch.
type FLOPsEstimate struct {
	ForwardFLOPs       int64 // FLOPs per forward pass
	TrainingFLOPs      int64 // FLOPs per training step (3x forward)
	FLOPsPerToken      int64 // TrainingFLOPs / batch_tokens
	ParamCount         int64 // Unique parameter count after sharing
	ExpandedParamCount int64 // Parameter count with recurrent/shared blocks expanded
}

// EstimateFLOPs returns an analytical FLOPs estimate for one configured batch.
func EstimateFLOPs(cfg *ArchConfig) FLOPsEstimate {
	if cfg == nil || cfg.ModelDim <= 0 || cfg.VocabSize <= 0 || cfg.SeqLen <= 0 || cfg.Training.BatchTokens <= 0 {
		return FLOPsEstimate{}
	}
	paramCount, expandedParamCount, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		return FLOPsEstimate{}
	}

	B := cfg.Training.BatchTokens / cfg.SeqLen
	if B <= 0 {
		return FLOPsEstimate{}
	}
	T := cfg.SeqLen
	D := cfg.ModelDim
	V := cfg.VocabSize
	ffn := ffnDim(D, cfg.MLPMult)

	forward := int64(0)
	for _, block := range cfg.Blocks {
		forward += estimateBlockFLOPs(block, B, T, D, V, ffn, cfg.MLPMult, cfg.BlockScales, cfg.ResidMix)
	}

	// Embedding lookup is indexing only; LM head projection produces logits.
	forward += 2 * i64(B) * i64(T) * i64(D) * i64(V)

	training := 3 * forward
	perToken := int64(0)
	if cfg.Training.BatchTokens > 0 {
		perToken = training / int64(cfg.Training.BatchTokens)
	}
	return FLOPsEstimate{
		ForwardFLOPs:       forward,
		TrainingFLOPs:      training,
		FLOPsPerToken:      perToken,
		ParamCount:         paramCount,
		ExpandedParamCount: expandedParamCount,
	}
}

func ParameterCountsFromConfig(cfg *ArchConfig) (int64, int64, error) {
	if cfg == nil {
		return 0, 0, fmt.Errorf("nil config")
	}

	uniqueRefs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return 0, 0, err
	}
	uniqueShapes, err := collectWeightShapesWithRefs(
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
		uniqueRefs,
	)
	if err != nil {
		return 0, 0, err
	}

	expandedShapes, err := collectWeightShapesWithRefs(
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
		identityWeightRefs(cfg.Blocks),
	)
	if err != nil {
		return 0, 0, err
	}

	return countWeightMetaElements(uniqueShapes), countWeightMetaElements(expandedShapes), nil
}

func countWeightMetaElements(metas []WeightMeta) int64 {
	total := int64(0)
	for _, meta := range metas {
		elements := int64(1)
		for _, dim := range meta.Shape {
			elements *= int64(dim)
		}
		total += elements
	}
	return total
}

func estimateBlockFLOPs(block BlockSpec, B, T, D, V, ffn int, mlpMult float64, blockScales, residMix bool) int64 {
	switch strings.ToLower(strings.TrimSpace(block.Type)) {
	case "plain":
		return estimatePlainBlockFLOPs(block, B, T, D, ffn)
	case "swiglu":
		return estimateSwiGLUBlockFLOPs(B, T, D, ffn)
	case "mamba":
		inner := block.InnerDim
		if inner <= 0 {
			inner = D
		}
		return 2 * i64(B) * i64(T) * i64(D) * i64(inner) * 4
	default:
		return estimateWeightShapeFLOPs(block, B, T, D, V, mlpMult, blockScales, residMix)
	}
}

func estimatePlainBlockFLOPs(block BlockSpec, B, T, D, ffn int) int64 {
	total := int64(0)
	heads := block.Heads
	if heads <= 0 {
		heads = 4
	}
	if heads <= 0 || D%heads != 0 {
		return total
	}
	headDim := D / heads
	kvHeads := block.KVHeads
	if kvHeads <= 0 {
		kvHeads = heads
	}

	if !block.SkipAttention {
		switch {
		case block.KVSource > 0:
			total += 2 * i64(B) * i64(T) * i64(D) * i64(D) // Q + O only
		case kvHeads < heads:
			total += 2 * i64(B) * i64(T) * i64(D) * (i64(D) + 2*i64(D)*i64(kvHeads)/i64(heads))
		default:
			total += 3 * 2 * i64(B) * i64(T) * i64(D) * i64(D)
		}
		total += 2 * i64(B) * i64(heads) * i64(T) * i64(T) * i64(headDim)
		total += 2 * i64(B) * i64(heads) * i64(T) * i64(T) * i64(headDim)
		total += 2 * i64(B) * i64(T) * i64(D) * i64(D)
	}

	total += 2 * 2 * i64(B) * i64(T) * i64(D) * i64(ffn)
	return total
}

func estimateSwiGLUBlockFLOPs(B, T, D, ffn int) int64 {
	gateUp := 2 * 2 * i64(B) * i64(T) * i64(D) * i64(ffn)
	down := 2 * i64(B) * i64(T) * i64(ffn) * i64(D)
	return gateUp + down
}

func estimateWeightShapeFLOPs(block BlockSpec, B, T, D, V int, mlpMult float64, blockScales, residMix bool) int64 {
	metas, err := blockWeightShapes(block, D, T, B, V, mlpMult, blockScales, residMix)
	if err != nil {
		return 0
	}
	total := int64(0)
	for _, meta := range metas {
		if len(meta.Shape) != 2 {
			continue
		}
		elements := int64(1)
		for _, dim := range meta.Shape {
			elements *= int64(dim)
		}
		total += 2 * i64(B) * i64(T) * elements
	}
	return total
}

func i64(v int) int64 {
	return int64(v)
}
