package arch

import "fmt"

// Thin dispatch wrappers around per-block and per-stream IR emission.

func emitStreamIR(prog *Program, specs []BlockSpec, stream, original string, wi, D, T, B, V int, opIdx *int, mlpMult float64, blockScales, residMix bool) (int, error) {
	return emitStreamIRWithDropout(prog, specs, stream, original, wi, D, T, B, V, opIdx, mlpMult, blockScales, residMix, 0, 0)
}

func emitStreamIRWithDropout(prog *Program, specs []BlockSpec, stream, original string, wi, D, T, B, V int, opIdx *int, mlpMult float64, blockScales, residMix bool, dropout, attnDropout float32) (int, error) {
	kvCache := make(map[int]BlockKVOutputs, len(specs))
	for i, spec := range specs {
		var err error
		if needsResidMix(spec, residMix) {
			wi = applyResidMixIR(prog, stream, original, wi, D, *opIdx)
		}
		wi, err = emitBlockIRWithDropout(prog, spec, stream, wi, D, T, B, V, *opIdx, i, nil, kvCache, mlpMult, blockScales, dropout, attnDropout)
		if err != nil {
			return wi, err
		}
		(*opIdx)++
	}
	return wi, nil
}

func emitSequentialBlockWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, blockIdx int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix bool, dropout, attnDropout float32, norm NormSpec, normPlacement string, ffnInternalNorm bool, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, layerAgg *layerAggregationBuildState, segmentMask bool) (int, error) {
	spec := specs[blockIdx]
	blockWI := wi
	originalBlock := refs[blockIdx] == blockIdx
	if !originalBlock {
		blockWI = weightStarts[refs[blockIdx]]
		if blockWI < 0 {
			return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", blockIdx)
		}
	}

	bodyWI := blockWI
	if needsResidMix(spec, residMix) {
		bodyWI = applyResidMixIR(prog, stream, original, bodyWI, D, *opIdx)
	}
	nextWI, err := emitBlockIRWithDropoutOptions(prog, spec, stream, bodyWI, D, T, B, V, *opIdx, blockIdx, streamSeqLens, kvCache, mlpMult, blockScales, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, positionalEmbedding, sharedRel, layerAgg, segmentMask)
	if err != nil {
		return wi, err
	}

	weightStarts[blockIdx] = blockWI
	if originalBlock {
		return nextWI, nil
	}
	return wi, nil
}

// emitBlockIR dispatches a single block emission.
// streamSeqLens maps stream names to their sequence lengths (used by cross_attention).
func emitBlockIR(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, streamSeqLens map[string]int, mlpMult float64, blockScales bool) (int, error) { //nolint:unparam // B is fixed at IR build time by design
	return emitBlockIRWithDropout(prog, spec, stream, wi, D, T, B, V, idx, idx, streamSeqLens, nil, mlpMult, blockScales, 0, 0)
}

func emitBlockIRWithDropout(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx, blockIndex int, streamSeqLens map[string]int, kvCache map[int]BlockKVOutputs, mlpMult float64, blockScales bool, dropout, attnDropout float32) (int, error) {
	return emitBlockIRWithDropoutOptions(prog, spec, stream, wi, D, T, B, V, idx, blockIndex, streamSeqLens, kvCache, mlpMult, blockScales, dropout, attnDropout, defaultNormSpec(), NormPlacementPre, false, PositionalEmbeddingRope, sharedRelativeAttentionPlan{WeightIndex: -1}, nil, false)
}

func emitBlockIRWithDropoutOptions(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx, blockIndex int, streamSeqLens map[string]int, kvCache map[int]BlockKVOutputs, mlpMult float64, blockScales bool, dropout, attnDropout float32, norm NormSpec, normPlacement string, ffnInternalNorm bool, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, layerAgg *layerAggregationBuildState, segmentMask bool) (int, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return wi, err
	}
	if reg.Emitter == nil {
		return wi, fmt.Errorf("block type %q has no emitter", spec.Type)
	}
	return reg.Emitter(prog, spec, stream, wi, D, T, B, V, idx, EmitOptions{
		StreamSeqLens:       streamSeqLens,
		MLPMult:             mlpMult,
		BlockScales:         blockScales,
		Dropout:             dropout,
		AttnDropout:         attnDropout,
		Norm:                normSpecOrDefault(norm),
		NormPlacement:       normPlacementOrDefault(normPlacement),
		FFNInternalNorm:     ffnInternalNorm,
		PositionalEmbedding: normalizePositionalEmbedding(positionalEmbedding),
		BlockIndex:          blockIndex,
		KVCache:             kvCache,
		SegmentMask:         segmentMask,
		sharedRelative:      sharedRel,
		layerAgg:            layerAgg,
	})
}
