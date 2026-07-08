package arch

import "fmt"

type parallelResidualPlan struct {
	starts       []bool
	seconds      []bool
	groupEnds    []int
	memberStarts []int
	any          bool
}

func newParallelResidualPlan(specs []BlockSpec, global bool) (parallelResidualPlan, error) {
	plan := parallelResidualPlan{
		starts:       make([]bool, len(specs)),
		seconds:      make([]bool, len(specs)),
		groupEnds:    make([]int, len(specs)),
		memberStarts: make([]int, len(specs)),
	}
	for i := range plan.groupEnds {
		plan.groupEnds[i] = -1
		plan.memberStarts[i] = -1
	}
	if hasExplicitParallelGroup(specs) && global {
		return plan, fmt.Errorf("parallel_group cannot be combined with top-level parallel_residual")
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
			plan.markGroup(i, 2)
		}
		return plan, nil
	}

	for i := range specs {
		if plan.memberStarts[i] >= 0 {
			if specs[i].ParallelGroup > 0 {
				return plan, fmt.Errorf("parallel_group for block %d overlaps group starting at block %d", i, plan.memberStarts[i])
			}
			if specs[i].ParallelResidual != nil && *specs[i].ParallelResidual {
				return plan, fmt.Errorf("parallel_residual block %d overlaps pair starting at block %d", i, plan.memberStarts[i])
			}
			continue
		}
		if specs[i].ParallelGroup > 0 {
			if err := validateParallelGroup(specs, i, specs[i].ParallelGroup); err != nil {
				return plan, err
			}
			plan.markGroup(i, specs[i].ParallelGroup)
			continue
		}
		if specs[i].ParallelResidual == nil || !*specs[i].ParallelResidual {
			continue
		}
		if err := validateParallelResidualPair(specs, i); err != nil {
			return plan, err
		}
		if i+1 < len(specs) && specs[i+1].ParallelResidual != nil && *specs[i+1].ParallelResidual {
			return plan, fmt.Errorf("parallel_residual block %d overlaps pair starting at block %d", i+1, i)
		}
		if i+1 < len(specs) && specs[i+1].ParallelGroup > 0 {
			return plan, fmt.Errorf("parallel_group for block %d overlaps pair starting at block %d", i+1, i)
		}
		plan.markGroup(i, 2)
	}
	return plan, nil
}

func hasExplicitParallelGroup(specs []BlockSpec) bool {
	for _, spec := range specs {
		if spec.ParallelGroup > 0 {
			return true
		}
	}
	return false
}

func (p *parallelResidualPlan) markGroup(start, length int) {
	end := start + length
	p.starts[start] = true
	p.groupEnds[start] = end
	for i := start; i < end; i++ {
		p.memberStarts[i] = start
		if i > start {
			p.seconds[i] = true
		}
	}
	p.any = true
}

func (p parallelResidualPlan) startsAt(i int) bool {
	return i >= 0 && i < len(p.starts) && p.starts[i]
}

func (p parallelResidualPlan) secondAt(i int) bool {
	return i >= 0 && i < len(p.seconds) && p.seconds[i]
}

func (p parallelResidualPlan) groupEndAt(i int) int {
	if !p.startsAt(i) {
		return -1
	}
	return p.groupEnds[i]
}

func (p parallelResidualPlan) groupLenAt(i int) int {
	end := p.groupEndAt(i)
	if end < 0 {
		return 0
	}
	return end - i
}

func (p parallelResidualPlan) memberStartAt(i int) int {
	if i < 0 || i >= len(p.memberStarts) {
		return -1
	}
	return p.memberStarts[i]
}

func (p parallelResidualPlan) anyInRange(start, end int) bool {
	for i := start; i < end && i < len(p.starts); i++ {
		if p.memberStartAt(i) >= 0 {
			return true
		}
	}
	return false
}

func (p parallelResidualPlan) splitsRange(start, end int) bool {
	if start < end && p.memberStartAt(start) >= 0 && !p.startsAt(start) {
		return true
	}
	if end > start {
		last := end - 1
		groupStart := p.memberStartAt(last)
		if groupStart >= 0 && p.groupEndAt(groupStart) > end {
			return true
		}
	}
	return false
}

func validateParallelResidualPair(specs []BlockSpec, start int) error {
	if start+1 >= len(specs) {
		return fmt.Errorf("parallel_residual requires blocks[%d] to be followed by swiglu or geglu, or moe", start)
	}
	firstType := blockTypeKey(specs[start])
	if firstType != "plain" && firstType != "gated_deltanet" {
		return fmt.Errorf("parallel_residual requires blocks[%d].type=plain or gated_deltanet (got %q)", start, specs[start].Type)
	}
	if firstType == "plain" && specs[start].KVSource > 0 {
		return fmt.Errorf("parallel_residual does not support blocks[%d].kv_source=%d", start, specs[start].KVSource)
	}
	if !isParallelResidualFFNSecond(specs[start+1]) {
		return fmt.Errorf("parallel_residual requires blocks[%d].type=swiglu or geglu, or moe (got %q)", start+1, specs[start+1].Type)
	}
	return nil
}

func validateParallelGroup(specs []BlockSpec, start, length int) error {
	if length < 2 {
		return fmt.Errorf("parallel_group on blocks[%d] must be >= 2", start)
	}
	end := start + length
	if end > len(specs) {
		return fmt.Errorf("parallel_group on blocks[%d] length=%d extends past blocks length=%d", start, length, len(specs))
	}
	mixers := 0
	for i := start; i < end; i++ {
		spec := specs[i]
		typ := blockTypeKey(spec)
		if isParallelGroupTokenMixer(spec) {
			if typ == "plain" && spec.KVSource > 0 {
				return fmt.Errorf("parallel_group does not support blocks[%d].kv_source=%d", i, spec.KVSource)
			}
			if typ == "plain" && spec.DifferentialAttention {
				return fmt.Errorf("parallel_group does not support blocks[%d].differential_attention in v1", i)
			}
			mixers++
			continue
		}
		if isParallelResidualFFNSecond(spec) {
			if i != end-1 {
				return fmt.Errorf("parallel_group blocks[%d].type=%q is an FFN branch and must be the final group member", i, spec.Type)
			}
			continue
		}
		return fmt.Errorf("parallel_group blocks[%d].type=%q is unsupported in v1", i, spec.Type)
	}
	if mixers < 2 {
		return fmt.Errorf("parallel_group on blocks[%d] requires at least two token-mixer branches", start)
	}
	return nil
}

func isParallelGroupTokenMixer(spec BlockSpec) bool {
	switch blockTypeKey(spec) {
	case "plain", "gated_deltanet", "hgrn2":
		return true
	default:
		return false
	}
}

func isParallelResidualFFNSecond(spec BlockSpec) bool {
	switch blockTypeKey(spec) {
	case "swiglu", "geglu", "moe":
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
	if pairedSecond {
		n -= parallelFollowerOmittedWeightCount(spec, residMix)
	}
	return n, nil
}

func parallelFollowerOmittedWeightCount(spec BlockSpec, residMix bool) int {
	omitted := 0
	if needsResidMix(spec, residMix) {
		omitted++
	}
	switch blockTypeKey(spec) {
	case "plain", "gated_deltanet", "hgrn2", "swiglu", "geglu", "moe":
		omitted++
	}
	return omitted
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
	if plan.splitsRange(start, end) {
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

func emitParallelBlockGroupWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, blockIdx int, groupLen int, stream, original string, wi, D, T, B int, opIdx *int, mlpMult float64, blockScales, residMix bool, dropout, attnDropout float32, backout *backoutBuildPlan, norm NormSpec, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, layerAgg *layerAggregationBuildState, segmentMask bool) (int, error) {
	if groupLen < 2 {
		return wi, fmt.Errorf("parallel_residual group at block %d must have at least two members", blockIdx)
	}
	groupEnd := blockIdx + groupLen
	branchWIs := make([]int, groupLen)
	for offset := 0; offset < groupLen; offset++ {
		i := blockIdx + offset
		spec := specs[i]
		blockWI := wi
		originalBlock := refs[i] == i
		if !originalBlock {
			blockWI = weightStarts[refs[i]]
			if blockWI < 0 {
				return wi, fmt.Errorf("weight sharing for block[%d] references block without emitted weights", i)
			}
		}
		branchWIs[offset] = blockWI
		if originalBlock {
			weightStarts[i] = blockWI
			n, err := parallelBlockWeightCount(spec, offset > 0, blockScales, residMix)
			if err != nil {
				return wi, err
			}
			wi += n
		}
	}

	firstSpec := specs[blockIdx]
	bodyWI := branchWIs[0]
	if needsResidMix(firstSpec, residMix) {
		bodyWI = applyResidMixIR(prog, stream, original, bodyWI, D, *opIdx)
	}
	xNorm := tmpName(stream+"_parallel", *opIdx) + "_x_norm"
	prog.RMSNorm(stream, weightName(bodyWI), xNorm, 1e-5)
	bodyWI++
	branchWIs[0] = bodyWI

	explicitGroup := firstSpec.ParallelGroup > 0
	groupState := ""
	for offset := 0; offset < groupLen; offset++ {
		i := blockIdx + offset
		idx := *opIdx
		if explicitGroup {
			idx += offset
		}
		if offset == 0 {
			var err error
			groupState, _, err = emitParallelFirstBranchStateIR(prog, specs[i], stream, xNorm, branchWIs[offset], D, T, B, idx, mlpMult, blockScales, dropout, attnDropout, norm, positionalEmbedding, sharedRel, segmentMask)
			if err != nil {
				return wi, err
			}
			continue
		}
		delta, _, err := emitParallelBranchDeltaIR(prog, specs[i], stream, xNorm, branchWIs[offset], D, T, B, idx, mlpMult, blockScales, dropout, attnDropout, norm, positionalEmbedding, sharedRel, segmentMask)
		if err != nil {
			return wi, err
		}
		next := stream
		if offset < groupLen-1 {
			next = tmpName(stream+"_parallel_sum", *opIdx) + fmt.Sprintf("_%d", offset)
		}
		prog.Add(groupState, delta, next)
		groupState = next
	}
	if backout != nil {
		for i := blockIdx; i < groupEnd; i++ {
			backout.captureAfterBlock(prog, i, stream)
		}
	}
	if layerAgg != nil {
		for i := blockIdx; i < groupEnd; i++ {
			layerAgg.apply(prog, stream)
		}
	}
	*opIdx += groupLen
	return wi, nil
}

func emitParallelFirstBranchStateIR(prog *Program, spec BlockSpec, x, xNorm string, wi, D, T, B, idx int, mlpMult float64, blockScales bool, dropout, attnDropout float32, norm NormSpec, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, segmentMask bool) (string, int, error) {
	switch blockTypeKey(spec) {
	case "plain":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		return emitPlainAttentionParallelDeltaIRWithDropoutEx(prog, x, xNorm, wi, heads, spec.KVHeads, D, T, B, idx, mlpMult, blockScales, dropout, attnDropout, spec.QKGain, spec.QKNorm, spec.DifferentialAttention, spec.DifferentialLambdaInit, spec.RopeDims, spec.RopeConvention, spec.AttnBias, spec.AttnValueGate, spec.XSA, spec.SparseAttnGate, spec.WindowSize, spec.AttentionMask, spec.RelativeAttention, spec.RelativeAttentionWindow, spec.RelativeAttentionParameterization, spec.FFNActivation, norm, spec.FFNPreNorm, spec.FFNBias, positionalEmbedding, sharedRel, segmentMask)
	case "gated_deltanet", "hgrn2":
		prefix := tmpName(x+"_parallel_"+blockTypeKey(spec), idx)
		delta, nextWI, err := emitParallelBranchDeltaIR(prog, spec, x, xNorm, wi, D, T, B, idx, mlpMult, blockScales, dropout, attnDropout, norm, positionalEmbedding, sharedRel, segmentMask)
		if err != nil {
			return "", wi, err
		}
		state := prefix + "_state"
		prog.Add(x, delta, state)
		return state, nextWI, nil
	default:
		return "", wi, fmt.Errorf("parallel_group first block type %q is unsupported", spec.Type)
	}
}

func emitParallelBranchDeltaIR(prog *Program, spec BlockSpec, x, xNorm string, wi, D, T, B, idx int, mlpMult float64, blockScales bool, dropout, attnDropout float32, norm NormSpec, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, segmentMask bool) (string, int, error) {
	switch blockTypeKey(spec) {
	case "plain":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		state, nextWI, err := emitPlainAttentionParallelDeltaIRWithDropoutEx(prog, x, xNorm, wi, heads, spec.KVHeads, D, T, B, idx, mlpMult, blockScales, dropout, attnDropout, spec.QKGain, spec.QKNorm, spec.DifferentialAttention, spec.DifferentialLambdaInit, spec.RopeDims, spec.RopeConvention, spec.AttnBias, spec.AttnValueGate, spec.XSA, spec.SparseAttnGate, spec.WindowSize, spec.AttentionMask, spec.RelativeAttention, spec.RelativeAttentionWindow, spec.RelativeAttentionParameterization, spec.FFNActivation, norm, spec.FFNPreNorm, spec.FFNBias, positionalEmbedding, sharedRel, segmentMask)
		if err != nil {
			return "", wi, err
		}
		delta := state + "_delta"
		prog.Sub(state, x, delta)
		return delta, nextWI, nil
	case "gated_deltanet":
		prefix := tmpName(x+"_parallel_gated_deltanet", idx)
		return emitGatedDeltaNetParallelDeltaIR(prog, spec, xNorm, wi, T, B, prefix, blockScales)
	case "hgrn2":
		prefix := tmpName(x+"_parallel_hgrn2", idx)
		return emitHGRN2DeltaIR(prog, spec, xNorm, wi, D, T, B, prefix, blockScales)
	case "swiglu":
		delta, nextWI := emitSwiGLUParallelDeltaIRWithDropout(prog, xNorm, wi, idx, mlpMult, blockScales, dropout)
		return delta, nextWI, nil
	case "geglu":
		delta, nextWI := emitGEGLUParallelDeltaIRWithDropout(prog, xNorm, wi, idx, mlpMult, blockScales, dropout)
		return delta, nextWI, nil
	case "moe":
		return emitMoEParallelDeltaIRWithDropout(prog, spec, xNorm, wi, D, T, B, idx, mlpMult, blockScales, dropout)
	default:
		return "", wi, fmt.Errorf("parallel_group does not support block type %q", spec.Type)
	}
}

func emitSequentialRangeWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, start, end int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix, parallelResidual bool, dropout, attnDropout float32, backout *backoutBuildPlan, norm NormSpec, normPlacement string, ffnInternalNorm bool, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, layerAgg *layerAggregationBuildState, segmentMask bool) (int, error) {
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return wi, err
	}
	if plan.anyInRange(start, end) {
		if plan.splitsRange(start, end) {
			return wi, fmt.Errorf("parallel_residual block range [%d,%d) must not split a block pair", start, end)
		}
		if err := validateParallelResidualRefs(plan, refs); err != nil {
			return wi, err
		}
	}
	for i := start; i < end; {
		var err error
		if plan.startsAt(i) {
			groupLen := plan.groupLenAt(i)
			wi, err = emitParallelBlockGroupWithRecurrenceDropout(prog, specs, refs, weightStarts, i, groupLen, stream, original, wi, D, T, B, opIdx, mlpMult, blockScales, residMix, dropout, attnDropout, backout, norm, positionalEmbedding, sharedRel, layerAgg, segmentMask)
			if err != nil {
				return wi, err
			}
			i += groupLen
			continue
		}
		wi, err = emitSequentialBlockWithRecurrenceDropout(prog, specs, refs, weightStarts, kvCache, i, stream, original, wi, D, T, B, V, opIdx, streamSeqLens, mlpMult, blockScales, residMix, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, positionalEmbedding, sharedRel, layerAgg, segmentMask)
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

func emitSequentialOrderWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, order []int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix, parallelResidual bool, dropout, attnDropout float32, backout *backoutBuildPlan, norm NormSpec, normPlacement string, ffnInternalNorm bool, positionalEmbedding string, sharedRel sharedRelativeAttentionPlan, layerAgg *layerAggregationBuildState, segmentMask bool) (int, error) {
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
			groupLen := plan.groupLenAt(i)
			for offset := 1; offset < groupLen; offset++ {
				member := i + offset
				if pos+offset >= len(order) || order[pos+offset] != member {
					return wi, fmt.Errorf("recurrence activation execution order must keep parallel_residual group [%d,%d) together", i, i+groupLen)
				}
				if seen[member] {
					return wi, fmt.Errorf("recurrence activation execution order repeats block %d", member)
				}
				seen[member] = true
			}
			wi, err = emitParallelBlockGroupWithRecurrenceDropout(prog, specs, refs, weightStarts, i, groupLen, stream, original, wi, D, T, B, opIdx, mlpMult, blockScales, residMix, dropout, attnDropout, backout, norm, positionalEmbedding, sharedRel, layerAgg, segmentMask)
			if err != nil {
				return wi, err
			}
			pos += groupLen
			continue
		}
		wi, err = emitSequentialBlockWithRecurrenceDropout(prog, specs, refs, weightStarts, kvCache, i, stream, original, wi, D, T, B, V, opIdx, streamSeqLens, mlpMult, blockScales, residMix, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, positionalEmbedding, sharedRel, layerAgg, segmentMask)
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
