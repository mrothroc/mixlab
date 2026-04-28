package arch

import "fmt"

func validateParallelResidualBlocks(specs []BlockSpec) error {
	if len(specs)%2 != 0 {
		return fmt.Errorf("parallel_residual requires an even number of blocks")
	}
	for i := 0; i < len(specs); i += 2 {
		if blockTypeKey(specs[i]) != "plain" {
			return fmt.Errorf("parallel_residual requires blocks[%d].type=plain (got %q)", i, specs[i].Type)
		}
		if specs[i].KVSource > 0 {
			return fmt.Errorf("parallel_residual does not support blocks[%d].kv_source=%d", i, specs[i].KVSource)
		}
		if blockTypeKey(specs[i+1]) != "swiglu" {
			return fmt.Errorf("parallel_residual requires blocks[%d].type=swiglu (got %q)", i+1, specs[i+1].Type)
		}
	}
	return nil
}

func parallelBlockWeightCount(spec BlockSpec, blockIdx int, blockScales, residMix bool) (int, error) {
	n, err := BlockWeightCount(spec, blockScales, residMix)
	if err != nil {
		return 0, err
	}
	if blockIdx%2 == 1 && blockTypeKey(spec) == "swiglu" {
		n--
	}
	return n, nil
}

func countStreamWeightsWithRefsAndParallel(specs []BlockSpec, refs []int, blockScales, residMix, parallelResidual bool) (int, error) {
	if !parallelResidual {
		return countStreamWeightsWithRefs(specs, refs, blockScales, residMix)
	}
	if err := validateParallelResidualBlocks(specs); err != nil {
		return 0, err
	}
	total := 0
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		n, err := parallelBlockWeightCount(spec, i, blockScales, residMix)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countStreamWeightsWithRecurrenceAndParallel(specs []BlockSpec, recurrence []int, blockScales, residMix, parallelResidual bool) (int, error) {
	refs, err := normalizeWeightRefs(specs, recurrence)
	if err != nil {
		return 0, err
	}
	return countStreamWeightsWithRefsAndParallel(specs, refs, blockScales, residMix, parallelResidual)
}

func countBlockRangeWeightsWithRefsAndParallel(specs []BlockSpec, refs []int, start, end int, blockScales, residMix, parallelResidual bool) (int, error) {
	if !parallelResidual {
		return countBlockRangeWeightsWithRefs(specs, refs, start, end, blockScales, residMix)
	}
	if start%2 != 0 || end%2 != 0 {
		return 0, fmt.Errorf("parallel_residual block range [%d,%d) must align with block pairs", start, end)
	}
	total := 0
	for i := start; i < end; i++ {
		if refs[i] != i {
			continue
		}
		n, err := parallelBlockWeightCount(specs[i], i, blockScales, residMix)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countBlockRangeWeightsWithRecurrenceAndParallel(specs []BlockSpec, rec []int, start, end int, blockScales, residMix, parallelResidual bool) (int, error) {
	return countBlockRangeWeightsWithRefsAndParallel(specs, rec, start, end, blockScales, residMix, parallelResidual)
}

func emitParallelBlockPairWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, blockIdx int, stream, original string, wi, D, T, B int, opIdx *int, mlpMult float64, blockScales, residMix bool, dropout float32) (int, error) {
	plainSpec := specs[blockIdx]
	swigluSpec := specs[blockIdx+1]

	plainWI := wi
	plainOriginal := refs[blockIdx] == blockIdx
	if !plainOriginal {
		plainWI = weightStarts[refs[blockIdx]]
		if plainWI < 0 {
			return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", blockIdx)
		}
	}
	if plainOriginal {
		weightStarts[blockIdx] = plainWI
		n, err := parallelBlockWeightCount(plainSpec, blockIdx, blockScales, residMix)
		if err != nil {
			return wi, err
		}
		wi += n
	}

	swigluWI := wi
	swigluOriginal := refs[blockIdx+1] == blockIdx+1
	if !swigluOriginal {
		swigluWI = weightStarts[refs[blockIdx+1]]
		if swigluWI < 0 {
			return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", blockIdx+1)
		}
	}
	if swigluOriginal {
		weightStarts[blockIdx+1] = swigluWI
		n, err := parallelBlockWeightCount(swigluSpec, blockIdx+1, blockScales, residMix)
		if err != nil {
			return wi, err
		}
		wi += n
	}

	bodyWI := plainWI
	if needsResidMix(plainSpec, residMix) {
		bodyWI = applyResidMixIR(prog, stream, original, bodyWI, D, *opIdx)
	}
	xNorm := tmpName(stream+"_parallel", *opIdx) + "_x_norm"
	prog.RMSNorm(stream, weightName(bodyWI), xNorm, 1e-5)
	bodyWI++

	heads := plainSpec.Heads
	if heads <= 0 {
		heads = 4
	}
	plainState, _, err := emitPlainAttentionParallelDeltaIRWithDropout(prog, stream, xNorm, bodyWI, heads, plainSpec.KVHeads, D, T, B, *opIdx, mlpMult, blockScales, dropout, plainSpec.QKGain, plainSpec.SparseAttnGate)
	if err != nil {
		return wi, err
	}
	mlpDelta, _ := emitSwiGLUParallelDeltaIRWithDropout(prog, xNorm, swigluWI, *opIdx, mlpMult, blockScales, dropout)
	prog.Add(plainState, mlpDelta, stream)
	*opIdx += 2
	return wi, nil
}

func emitSequentialRangeWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, start, end int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix, parallelResidual bool, dropout float32) (int, error) {
	if parallelResidual {
		if start%2 != 0 || end%2 != 0 {
			return wi, fmt.Errorf("parallel_residual block range [%d,%d) must align with block pairs", start, end)
		}
		for i := start; i < end; i += 2 {
			var err error
			wi, err = emitParallelBlockPairWithRecurrenceDropout(prog, specs, refs, weightStarts, i, stream, original, wi, D, T, B, opIdx, mlpMult, blockScales, residMix, dropout)
			if err != nil {
				return wi, err
			}
		}
		return wi, nil
	}
	for i := start; i < end; i++ {
		var err error
		wi, err = emitSequentialBlockWithRecurrenceDropout(prog, specs, refs, weightStarts, kvCache, i, stream, original, wi, D, T, B, V, opIdx, streamSeqLens, mlpMult, blockScales, residMix, dropout)
		if err != nil {
			return wi, err
		}
		(*opIdx)++
	}
	return wi, nil
}
