package arch

import "fmt"

type parallelResidualPlan struct {
	starts  []bool
	seconds []bool
	any     bool
}

func newParallelResidualPlan(specs []BlockSpec, global bool) (parallelResidualPlan, error) {
	plan := parallelResidualPlan{
		starts:  make([]bool, len(specs)),
		seconds: make([]bool, len(specs)),
	}
	if global {
		if len(specs)%2 != 0 {
			return plan, fmt.Errorf("parallel_residual requires an even number of blocks")
		}
		for i := 0; i < len(specs); i += 2 {
			if specs[i+1].ParallelResidual != nil && *specs[i+1].ParallelResidual {
				return plan, fmt.Errorf("parallel_residual for block pair [%d,%d] must be set on blocks[%d]", i, i+1, i)
			}
			enabled := true
			if specs[i].ParallelResidual != nil {
				enabled = *specs[i].ParallelResidual
			}
			if !enabled {
				continue
			}
			if err := validateParallelResidualPair(specs, i); err != nil {
				return plan, err
			}
			plan.markPair(i)
		}
		return plan, nil
	}

	for i := range specs {
		if specs[i].ParallelResidual == nil || !*specs[i].ParallelResidual {
			continue
		}
		if plan.seconds[i] {
			return plan, fmt.Errorf("parallel_residual block %d overlaps pair starting at block %d", i, i-1)
		}
		if err := validateParallelResidualPair(specs, i); err != nil {
			return plan, err
		}
		if i+1 < len(specs) && specs[i+1].ParallelResidual != nil && *specs[i+1].ParallelResidual {
			return plan, fmt.Errorf("parallel_residual block %d overlaps pair starting at block %d", i+1, i)
		}
		plan.markPair(i)
	}
	return plan, nil
}

func (p *parallelResidualPlan) markPair(start int) {
	p.starts[start] = true
	p.seconds[start+1] = true
	p.any = true
}

func (p parallelResidualPlan) startsAt(i int) bool {
	return i >= 0 && i < len(p.starts) && p.starts[i]
}

func (p parallelResidualPlan) secondAt(i int) bool {
	return i >= 0 && i < len(p.seconds) && p.seconds[i]
}

func (p parallelResidualPlan) anyInRange(start, end int) bool {
	for i := start; i < end && i < len(p.starts); i++ {
		if p.startsAt(i) || p.secondAt(i) {
			return true
		}
	}
	return false
}

func validateParallelResidualPair(specs []BlockSpec, start int) error {
	if start+1 >= len(specs) {
		return fmt.Errorf("parallel_residual requires blocks[%d] to be followed by swiglu or geglu", start)
	}
	firstType := blockTypeKey(specs[start])
	if firstType != "plain" && firstType != "gated_deltanet" {
		return fmt.Errorf("parallel_residual requires blocks[%d].type=plain or gated_deltanet (got %q)", start, specs[start].Type)
	}
	if firstType == "plain" && specs[start].KVSource > 0 {
		return fmt.Errorf("parallel_residual does not support blocks[%d].kv_source=%d", start, specs[start].KVSource)
	}
	if !isParallelResidualGLUSecond(specs[start+1]) {
		return fmt.Errorf("parallel_residual requires blocks[%d].type=swiglu or geglu (got %q)", start+1, specs[start+1].Type)
	}
	return nil
}

func isParallelResidualGLUSecond(spec BlockSpec) bool {
	switch blockTypeKey(spec) {
	case "swiglu", "geglu":
		return true
	default:
		return false
	}
}

func validateParallelResidualRefs(plan parallelResidualPlan, refs []int) error {
	for i, ref := range refs {
		if ref == i {
			continue
		}
		if plan.secondAt(i) != plan.secondAt(ref) {
			return fmt.Errorf("parallel_residual block[%d] cannot share weights with block[%d] because their GLU norm layout differs", i, ref)
		}
	}
	return nil
}

func parallelBlockWeightCount(spec BlockSpec, pairedSecond bool, blockScales, residMix bool) (int, error) {
	n, err := BlockWeightCount(spec, blockScales, residMix)
	if err != nil {
		return 0, err
	}
	if pairedSecond && isParallelResidualGLUSecond(spec) {
		n--
	}
	return n, nil
}

func countStreamWeightsWithRefsAndParallel(specs []BlockSpec, refs []int, blockScales, residMix, parallelResidual bool) (int, error) {
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return 0, err
	}
	if !plan.any {
		return countStreamWeightsWithRefs(specs, refs, blockScales, residMix)
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return 0, err
	}
	total := 0
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		n, err := parallelBlockWeightCount(spec, plan.secondAt(i), blockScales, residMix)
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
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return 0, err
	}
	if !plan.anyInRange(start, end) {
		return countBlockRangeWeightsWithRefs(specs, refs, start, end, blockScales, residMix)
	}
	if plan.secondAt(start) || plan.startsAt(end-1) {
		return 0, fmt.Errorf("parallel_residual block range [%d,%d) must not split a block pair", start, end)
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return 0, err
	}
	total := 0
	for i := start; i < end; i++ {
		if refs[i] != i {
			continue
		}
		n, err := parallelBlockWeightCount(specs[i], plan.secondAt(i), blockScales, residMix)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countBlockRangeWeightsWithRecurrenceAndParallel(specs []BlockSpec, rec []int, start, end int, blockScales, residMix, parallelResidual bool) (int, error) {
	refs, err := normalizeWeightRefs(specs, rec)
	if err != nil {
		return 0, err
	}
	return countBlockRangeWeightsWithRefsAndParallel(specs, refs, start, end, blockScales, residMix, parallelResidual)
}

func emitParallelBlockPairWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, blockIdx int, stream, original string, wi, D, T, B int, opIdx *int, mlpMult float64, blockScales, residMix bool, dropout float32, backout *backoutBuildPlan) (int, error) {
	firstSpec := specs[blockIdx]
	gluSpec := specs[blockIdx+1]

	firstWI := wi
	firstOriginal := refs[blockIdx] == blockIdx
	if !firstOriginal {
		firstWI = weightStarts[refs[blockIdx]]
		if firstWI < 0 {
			return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", blockIdx)
		}
	}
	if firstOriginal {
		weightStarts[blockIdx] = firstWI
		n, err := parallelBlockWeightCount(firstSpec, false, blockScales, residMix)
		if err != nil {
			return wi, err
		}
		wi += n
	}

	gluWI := wi
	gluOriginal := refs[blockIdx+1] == blockIdx+1
	if !gluOriginal {
		gluWI = weightStarts[refs[blockIdx+1]]
		if gluWI < 0 {
			return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", blockIdx+1)
		}
	}
	if gluOriginal {
		weightStarts[blockIdx+1] = gluWI
		n, err := parallelBlockWeightCount(gluSpec, true, blockScales, residMix)
		if err != nil {
			return wi, err
		}
		wi += n
	}

	bodyWI := firstWI
	if needsResidMix(firstSpec, residMix) {
		bodyWI = applyResidMixIR(prog, stream, original, bodyWI, D, *opIdx)
	}
	xNorm := tmpName(stream+"_parallel", *opIdx) + "_x_norm"
	prog.RMSNorm(stream, weightName(bodyWI), xNorm, 1e-5)
	bodyWI++

	var firstState string
	switch blockTypeKey(firstSpec) {
	case "plain":
		heads := firstSpec.Heads
		if heads <= 0 {
			heads = 4
		}
		var err error
		firstState, _, err = emitPlainAttentionParallelDeltaIRWithDropout(prog, stream, xNorm, bodyWI, heads, firstSpec.KVHeads, D, T, B, *opIdx, mlpMult, blockScales, dropout, firstSpec.QKGain, firstSpec.RopeDims, firstSpec.XSA, firstSpec.SparseAttnGate, firstSpec.WindowSize, firstSpec.AttentionMask, firstSpec.RelativeAttention, firstSpec.RelativeAttentionWindow)
		if err != nil {
			return wi, err
		}
	case "gated_deltanet":
		prefix := tmpName(stream+"_parallel_gated_deltanet", *opIdx)
		var err error
		firstState, _, err = emitGatedDeltaNetParallelStateIR(prog, firstSpec, stream, xNorm, bodyWI, T, B, prefix)
		if err != nil {
			return wi, err
		}
	default:
		return wi, fmt.Errorf("parallel_residual requires blocks[%d].type=plain or gated_deltanet (got %q)", blockIdx, firstSpec.Type)
	}
	var mlpDelta string
	switch blockTypeKey(gluSpec) {
	case "swiglu":
		mlpDelta, _ = emitSwiGLUParallelDeltaIRWithDropout(prog, xNorm, gluWI, *opIdx, mlpMult, blockScales, dropout)
	case "geglu":
		mlpDelta, _ = emitGEGLUParallelDeltaIRWithDropout(prog, xNorm, gluWI, *opIdx, mlpMult, blockScales, dropout)
	default:
		return wi, fmt.Errorf("parallel_residual requires blocks[%d].type=swiglu or geglu (got %q)", blockIdx+1, gluSpec.Type)
	}
	prog.Add(firstState, mlpDelta, stream)
	if backout != nil {
		backout.captureAfterBlock(prog, blockIdx, stream)
		backout.captureAfterBlock(prog, blockIdx+1, stream)
	}
	*opIdx += 2
	return wi, nil
}

func emitSequentialRangeWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, start, end int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix, parallelResidual bool, dropout float32, backout *backoutBuildPlan) (int, error) {
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return wi, err
	}
	if plan.anyInRange(start, end) {
		if plan.secondAt(start) || plan.startsAt(end-1) {
			return wi, fmt.Errorf("parallel_residual block range [%d,%d) must not split a block pair", start, end)
		}
		if err := validateParallelResidualRefs(plan, refs); err != nil {
			return wi, err
		}
	}
	for i := start; i < end; {
		var err error
		if plan.startsAt(i) {
			wi, err = emitParallelBlockPairWithRecurrenceDropout(prog, specs, refs, weightStarts, i, stream, original, wi, D, T, B, opIdx, mlpMult, blockScales, residMix, dropout, backout)
			if err != nil {
				return wi, err
			}
			i += 2
			continue
		}
		wi, err = emitSequentialBlockWithRecurrenceDropout(prog, specs, refs, weightStarts, kvCache, i, stream, original, wi, D, T, B, V, opIdx, streamSeqLens, mlpMult, blockScales, residMix, dropout)
		if err != nil {
			return wi, err
		}
		if backout != nil {
			backout.captureAfterBlock(prog, i, stream)
		}
		i++
		(*opIdx)++
	}
	return wi, nil
}

func emitSequentialOrderWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, order []int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix, parallelResidual bool, dropout float32, backout *backoutBuildPlan) (int, error) {
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return wi, err
	}
	if plan.any {
		if err := validateParallelResidualRefs(plan, refs); err != nil {
			return wi, err
		}
	}
	seen := make(map[int]bool, len(order))
	for pos := 0; pos < len(order); {
		i := order[pos]
		if i < 0 || i >= len(specs) {
			return wi, fmt.Errorf("recurrence activation execution order index %d out of range [0,%d)", i, len(specs))
		}
		if seen[i] {
			return wi, fmt.Errorf("recurrence activation execution order repeats block %d", i)
		}
		seen[i] = true
		if plan.secondAt(i) {
			return wi, fmt.Errorf("recurrence activation execution order cannot start at parallel_residual follower block %d", i)
		}
		if plan.startsAt(i) {
			if pos+1 >= len(order) || order[pos+1] != i+1 {
				return wi, fmt.Errorf("recurrence activation execution order must keep parallel_residual pair [%d,%d] together", i, i+1)
			}
			if seen[i+1] {
				return wi, fmt.Errorf("recurrence activation execution order repeats block %d", i+1)
			}
			seen[i+1] = true
			wi, err = emitParallelBlockPairWithRecurrenceDropout(prog, specs, refs, weightStarts, i, stream, original, wi, D, T, B, opIdx, mlpMult, blockScales, residMix, dropout, backout)
			if err != nil {
				return wi, err
			}
			pos += 2
			continue
		}
		wi, err = emitSequentialBlockWithRecurrenceDropout(prog, specs, refs, weightStarts, kvCache, i, stream, original, wi, D, T, B, V, opIdx, streamSeqLens, mlpMult, blockScales, residMix, dropout)
		if err != nil {
			return wi, err
		}
		if backout != nil {
			backout.captureAfterBlock(prog, i, stream)
		}
		pos++
		(*opIdx)++
	}
	return wi, nil
}
