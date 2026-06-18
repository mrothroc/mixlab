package arch

import (
	"fmt"
	"strings"
)

// BlockWeightCount returns the number of IR weight tensors consumed by a block.
func BlockWeightCount(spec BlockSpec, blockScales, residMix bool) (int, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return 0, err
	}
	if reg.WeightCount == nil {
		return 0, fmt.Errorf("block type %q has no weight counter", spec.Type)
	}
	return reg.WeightCount(spec, blockScales, residMix)
}

// blockTypeKey returns a canonical key for weight-tying purposes.
func blockTypeKey(spec BlockSpec) string {
	return blockTypeName(spec.Type)
}

func normalizePlainKVHeads(heads, kvHeads int) (int, error) {
	if heads <= 0 {
		return 0, fmt.Errorf("invalid attention head count H=%d", heads)
	}
	if kvHeads <= 0 {
		return heads, nil
	}
	if kvHeads > heads || heads%kvHeads != 0 {
		return 0, fmt.Errorf("invalid grouped attention dimensions H=%d KV=%d", heads, kvHeads)
	}
	return kvHeads, nil
}

func normalizeRecurrence(specs []BlockSpec, recurrence []int) ([]int, error) {
	if recurrence != nil && len(recurrence) != len(specs) {
		return nil, fmt.Errorf("recurrence length=%d must match blocks length=%d", len(recurrence), len(specs))
	}
	out := make([]int, len(specs))
	for i := range specs {
		ref := i
		if recurrence != nil {
			ref = recurrence[i]
		}
		if ref < 0 || ref >= len(specs) {
			return nil, fmt.Errorf("recurrence[%d]=%d out of range [0,%d)", i, ref, len(specs))
		}
		if ref > i {
			return nil, fmt.Errorf("recurrence[%d]=%d is a forward reference", i, ref)
		}
		if blockTypeKey(specs[i]) != blockTypeKey(specs[ref]) {
			return nil, fmt.Errorf("recurrence[%d]=%d type mismatch: blocks[%d].type=%q blocks[%d].type=%q", i, ref, i, specs[i].Type, ref, specs[ref].Type)
		}
		out[i] = ref
	}
	return out, nil
}

func normalizeWeightRefs(specs []BlockSpec, recurrence []int) ([]int, error) {
	rec, err := normalizeRecurrence(specs, recurrence)
	if err != nil {
		return nil, err
	}

	refs := make([]int, len(specs))
	groupStarts := make(map[string]int)
	for i, spec := range specs {
		ref := rec[i]
		if ref != i {
			ref = refs[ref]
		}

		group := strings.TrimSpace(spec.WeightGroup)
		if group == "" {
			refs[i] = ref
			continue
		}

		if groupRef, ok := groupStarts[group]; ok {
			if ref != i && ref != groupRef {
				return nil, fmt.Errorf("block[%d] weight_group=%q conflicts with recurrence[%d]=%d", i, group, i, rec[i])
			}
			ref = groupRef
		} else {
			groupStarts[group] = ref
		}
		refs[i] = ref
	}
	return refs, nil
}

func uniqueRecurrenceExecutionOrder(specs []BlockSpec, recurrence []int) ([]int, error) {
	if recurrence == nil {
		return nil, nil
	}
	rec, err := normalizeRecurrence(specs, recurrence)
	if err != nil {
		return nil, err
	}
	roots := make([]int, len(specs))
	for i := range specs {
		ref := rec[i]
		if ref != i {
			ref = roots[ref]
		}
		roots[i] = ref
	}

	seen := make(map[int]bool, len(roots))
	order := make([]int, 0, len(roots))
	for _, ref := range roots {
		if seen[ref] {
			continue
		}
		seen[ref] = true
		order = append(order, ref)
	}
	return order, nil
}

func identityWeightRefs(specs []BlockSpec) []int {
	refs := make([]int, len(specs))
	for i := range specs {
		refs[i] = i
	}
	return refs
}

func countStreamWeightsWithRefs(specs []BlockSpec, refs []int, blockScales, residMix bool) (int, error) {
	total := 0
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		n, err := BlockWeightCount(spec, blockScales, residMix)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countBlockRangeWeightsWithRefs(specs []BlockSpec, refs []int, start, end int, blockScales, residMix bool) (int, error) {
	total := 0
	for i := start; i < end; i++ {
		if refs[i] != i {
			continue
		}
		n, err := BlockWeightCount(specs[i], blockScales, residMix)
		if err != nil {
			return 0, err
		}
		total += n
	}
	return total, nil
}

func countWeightsWithBigramRecurrenceParallelHeadLayout(
	modelDim int,
	mlpMult float64,
	reserveHead bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) (int, error) {
	_ = mlpMult
	total := fixedWeightCountWithHead(reserveHead)
	total += bigramWeightCount(modelDim, bigramVocabSize, bigramDim)
	sharedRelCount, err := sharedRelativeAttentionWeightCount(blocks)
	if err != nil {
		return 0, err
	}
	total += sharedRelCount

	plan, err := newParallelResidualPlan(blocks, parallelResidual)
	if err != nil {
		return 0, err
	}
	if plan.any && unet {
		return 0, fmt.Errorf("parallel_residual is not supported with unet")
	}
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return 0, fmt.Errorf("blocks: %w", err)
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return 0, fmt.Errorf("blocks: %w", err)
	}
	if unet {
		numEncoder, numSkip := unetLayout(len(blocks))
		n, err := countBlockRangeWeightsWithRefsAndParallel(blocks, refs, 0, numEncoder, blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n + numSkip
		n, err = countBlockRangeWeightsWithRefsAndParallel(blocks, refs, numEncoder, len(blocks), blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n
	} else {
		n, err := countStreamWeightsWithRefsAndParallel(blocks, refs, blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n
	}

	return total, nil
}

func countWeightsWithNgramsRecurrenceAndParallel(
	modelDim int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) (int, error) {
	total, err := countWeightsWithBigramRecurrenceParallelHeadLayout(modelDim, mlpMult, !tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, blocks, recurrence)
	if err != nil {
		return 0, err
	}
	return total + trigramWeightCount(modelDim, trigramVocabSize, trigramDim), nil
}

func countWeightsWithFeaturesRecurrenceParallelHeadLayoutNorm(
	modelDim int,
	vocabSize int,
	seqLen int,
	mlpMult float64,
	reserveHead bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	charVocabSize, charDim int,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	recurrence []int,
	norm NormSpec,
	normPlacement string,
	ffnInternalNorm bool,
) (int, error) {
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return 0, fmt.Errorf("blocks: %w", err)
	}
	metas, err := collectWeightShapesWithRefsHeadLayoutFeaturesNorm(
		modelDim, vocabSize, seqLen, mlpMult, reserveHead, blockScales, residMix, unet, parallelResidual,
		charVocabSize, charDim, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim,
		blocks, refs, norm, normPlacement, ffnInternalNorm,
	)
	if err != nil {
		return 0, err
	}
	return len(metas), nil
}

// CountWeightsWithNgramsRecurrenceAndParallel returns the IR weight tensor
// count, including optional bigram/trigram embeddings, sequential block weight
// tying, and parallel residual block pairs.
func CountWeightsWithNgramsRecurrenceAndParallel(
	modelDim int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) (int, error) {
	return countWeightsWithNgramsRecurrenceAndParallel(modelDim, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, recurrence)
}

func unetLayout(numBlocks int) (numEncoder, numSkip int) {
	numEncoder = numBlocks / 2
	numDecoder := numBlocks - numEncoder
	numSkip = numEncoder
	if numDecoder < numSkip {
		numSkip = numDecoder
	}
	return numEncoder, numSkip
}

func fixedWeightCountWithHead(reserveHead bool) int {
	if !reserveHead {
		return 2 // embed + final_norm
	}
	return 3 // embed + head + final_norm
}

func fixedWeightCountWithHeadAndNorm(reserveHead bool, norm NormSpec) int {
	total := 1 // embed
	if reserveHead {
		total++
	}
	return total + len(normWeights("final_norm", 1, normSpecOrDefault(norm)))
}

func finalNormWeightIndexWithHeadAndNorm(reserveHead bool, norm NormSpec) int {
	_ = norm
	if !reserveHead {
		return 1
	}
	return 2
}

func needsResidMix(spec BlockSpec, residMix bool) bool {
	return residMix && strings.EqualFold(strings.TrimSpace(spec.Type), "plain")
}

func applyResidMixIR(prog *Program, x, x0 string, wi, D, idx int) int {
	prefix := tmpName(x+"_resid_mix", idx)
	mix0Row := prefix + "_mix0_row"
	mix0 := prefix + "_mix0"
	mix1Row := prefix + "_mix1_row"
	mix1 := prefix + "_mix1"
	xMix := prefix + "_x_mix"
	x0Mix := prefix + "_x0_mix"

	prog.Slice(weightName(wi), 0, 1, 1, 0, mix0Row)
	prog.Reshape(mix0Row, []int{D}, mix0)
	prog.Slice(weightName(wi), 1, 2, 1, 0, mix1Row)
	prog.Reshape(mix1Row, []int{D}, mix1)
	prog.Mul(x, mix0, xMix)
	prog.Mul(x0, mix1, x0Mix)
	prog.Add(xMix, x0Mix, x)
	return wi + 1
}

// emitStreamIR emits all blocks in a stream against the named hidden state.
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

func emitSequentialBlockWithRecurrenceDropout(prog *Program, specs []BlockSpec, refs []int, weightStarts []int, kvCache map[int]BlockKVOutputs, blockIdx int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix bool, dropout, attnDropout float32, norm NormSpec, normPlacement string, ffnInternalNorm bool, sharedRel sharedRelativeAttentionPlan) (int, error) {
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
	nextWI, err := emitBlockIRWithDropoutOptions(prog, spec, stream, bodyWI, D, T, B, V, *opIdx, blockIdx, streamSeqLens, kvCache, mlpMult, blockScales, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, sharedRel)
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
	return emitBlockIRWithDropoutOptions(prog, spec, stream, wi, D, T, B, V, idx, blockIndex, streamSeqLens, kvCache, mlpMult, blockScales, dropout, attnDropout, defaultNormSpec(), NormPlacementPre, false, sharedRelativeAttentionPlan{WeightIndex: -1})
}

func emitBlockIRWithDropoutOptions(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx, blockIndex int, streamSeqLens map[string]int, kvCache map[int]BlockKVOutputs, mlpMult float64, blockScales bool, dropout, attnDropout float32, norm NormSpec, normPlacement string, ffnInternalNorm bool, sharedRel sharedRelativeAttentionPlan) (int, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return wi, err
	}
	if reg.Emitter == nil {
		return wi, fmt.Errorf("block type %q has no emitter", spec.Type)
	}
	return reg.Emitter(prog, spec, stream, wi, D, T, B, V, idx, EmitOptions{
		StreamSeqLens:   streamSeqLens,
		MLPMult:         mlpMult,
		BlockScales:     blockScales,
		Dropout:         dropout,
		AttnDropout:     attnDropout,
		Norm:            normSpecOrDefault(norm),
		NormPlacement:   normPlacementOrDefault(normPlacement),
		FFNInternalNorm: ffnInternalNorm,
		BlockIndex:      blockIndex,
		KVCache:         kvCache,
		sharedRelative:  sharedRel,
	})
}

// BuildIRProgram constructs a complete IR forward-pass program from a model
// configuration: embed -> blocks -> norm -> head -> loss.
func BuildIRProgram(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	logitSoftcap float32,
	blocks []BlockSpec,
) (*Program, error) {
	return BuildIRProgramWithBigram(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, 0, 0, logitSoftcap, blocks)
}

func BuildIRProgramWithRecurrence(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	logitSoftcap float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	return BuildIRProgramWithBigramAndRecurrence(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, 0, 0, logitSoftcap, blocks, recurrence)
}

// BuildIRProgramWithBigram constructs an IR program and optionally injects
// model-level bigram embeddings before the first block.
func BuildIRProgramWithBigram(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	logitSoftcap float32,
	blocks []BlockSpec,
) (*Program, error) {
	return BuildIRProgramWithBigramAndRecurrence(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, bigramVocabSize, bigramDim, logitSoftcap, blocks, nil)
}

// BuildIRProgramWithBigramAndRecurrence constructs an IR program and optionally
// injects model-level bigram embeddings before the first block. When recurrence
// is set, sequential blocks with recurrence[i] != i reuse the referenced
// block's weight span and do not advance the global weight index.
func BuildIRProgramWithBigramAndRecurrence(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	logitSoftcap float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	return BuildIRProgramWithBigramRecurrenceAndParallel(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, false, bigramVocabSize, bigramDim, logitSoftcap, blocks, recurrence)
}

func BuildIRProgramWithBigramRecurrenceParallelDropout(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	logitSoftcap float32,
	dropout float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	if dropout < 0 || dropout > 1 {
		return nil, fmt.Errorf("invalid dropout=%g", dropout)
	}
	return buildIRProgramWithDropout(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, logitSoftcap, dropout, blocks, recurrence)
}

// BuildIRProgramWithBigramRecurrenceAndParallel constructs an IR program and
// optionally injects model-level bigram embeddings, ties sequential block
// weights, and emits parallel residual block pairs.
func BuildIRProgramWithBigramRecurrenceAndParallel(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	logitSoftcap float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	return buildIRProgramWithDropout(modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, logitSoftcap, 0, blocks, recurrence)
}

func buildIRProgramWithDropout(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	logitSoftcap float32,
	dropout float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	return buildIRProgramWithDropoutAndNgrams(
		modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix,
		unet, parallelResidual, bigramVocabSize, bigramDim, 0, 0, logitSoftcap, dropout, dropout, blocks, recurrence,
	)
}

func buildIRProgramWithDropoutAndNgrams(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	logitSoftcap float32,
	dropout, attnDropout float32,
	blocks []BlockSpec,
	recurrence []int,
) (*Program, error) {
	return buildIRProgramWithDropoutNgramsAndOrder(
		modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix,
		unet, parallelResidual, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim,
		logitSoftcap, dropout, attnDropout, blocks, recurrence, nil, nil, !tieEmbeddings, tieEmbeddings,
		ObjectiveCausal,
	)
}

func buildIRProgramWithDropoutNgramsAndOrder(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	logitSoftcap float32,
	dropout, attnDropout float32,
	blocks []BlockSpec,
	recurrence []int,
	executionOrder []int,
	mtp *MTPSpec,
	reserveHead bool,
	useTiedHead bool,
	objective string,
) (*Program, error) {
	return buildIRProgramWithDropoutNgramsOrderAndSmear(
		modelDim, vocabSize, seqLen, batchSize, mlpMult, tieEmbeddings, blockScales, residMix,
		unet, parallelResidual, 0, 0, 0, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim,
		logitSoftcap, dropout, attnDropout, blocks, recurrence, executionOrder, mtp, reserveHead, useTiedHead,
		objective,
		objective,
		MLMHeadLinear,
		false,
		0,
		disabledSmearEmbeddingOptions(),
		nil,
		nil,
		nil,
		nil,
		defaultNormSpec(),
		NormPlacementPre,
		false,
	)
}

func buildIRProgramWithDropoutNgramsOrderAndSmear(
	modelDim, vocabSize, seqLen, batchSize int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	charVocabSize, charDim, charMaxPerToken int,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	logitSoftcap float32,
	dropout, attnDropout float32,
	blocks []BlockSpec,
	recurrence []int,
	executionOrder []int,
	mtp *MTPSpec,
	reserveHead bool,
	useTiedHead bool,
	objective string,
	rootObjective string,
	mlmHead string,
	firstByteMask bool,
	zLoss float64,
	smearOpts smearEmbeddingOptions,
	backoutSpec *BackoutSpec,
	distillation *DistillationSpec,
	data2vec *Data2VecSpec,
	hiddenCapture *data2VecHiddenCapture,
	norm NormSpec,
	normPlacement string,
	ffnInternalNorm bool,
) (*Program, error) {
	if mlpMult <= 0 {
		mlpMult = DefaultFFNMultiplier
	}
	if dropout < 0 || dropout > 1 {
		return nil, fmt.Errorf("invalid dropout=%g", dropout)
	}
	if attnDropout < 0 || attnDropout > 1 {
		return nil, fmt.Errorf("invalid attn_dropout=%g", attnDropout)
	}

	B := batchSize
	T := seqLen
	D := modelDim
	V := vocabSize
	objective = normalizeTrainingObjective(objective)
	rootObjective = normalizeTrainingObjective(rootObjective)
	mlmHead = normalizeMLMHead(mlmHead)
	blocks = resolveBlockAttentionMasksForObjective(blocks, objective, rootObjective)
	norm = normSpecOrDefault(norm)
	normPlacement = normPlacementOrDefault(normPlacement)

	if B <= 0 || T <= 0 {
		return nil, fmt.Errorf("invalid shape B=%d T=%d", B, T)
	}
	if D <= 0 {
		return nil, fmt.Errorf("invalid model_dim=%d", D)
	}
	if V <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", V)
	}
	if charVocabSize < 0 {
		return nil, fmt.Errorf("invalid char_vocab_size=%d", charVocabSize)
	}
	if charVocabSize > 0 && charVocabSize < 257 {
		return nil, fmt.Errorf("invalid char_vocab_size=%d", charVocabSize)
	}
	if charDim < 0 {
		return nil, fmt.Errorf("invalid char_dim=%d", charDim)
	}
	if charMaxPerToken < 0 {
		return nil, fmt.Errorf("invalid char_max_per_token=%d", charMaxPerToken)
	}
	if charVocabSize > 0 && charMaxPerToken == 0 {
		charMaxPerToken = 16
	}
	if bigramVocabSize < 0 {
		return nil, fmt.Errorf("invalid bigram_vocab_size=%d", bigramVocabSize)
	}
	if bigramVocabSize == 1 {
		return nil, fmt.Errorf("invalid bigram_vocab_size=%d", bigramVocabSize)
	}
	if bigramDim < 0 {
		return nil, fmt.Errorf("invalid bigram_dim=%d", bigramDim)
	}
	if trigramVocabSize < 0 {
		return nil, fmt.Errorf("invalid trigram_vocab_size=%d", trigramVocabSize)
	}
	if trigramVocabSize == 1 {
		return nil, fmt.Errorf("invalid trigram_vocab_size=%d", trigramVocabSize)
	}
	if trigramDim < 0 {
		return nil, fmt.Errorf("invalid trigram_dim=%d", trigramDim)
	}
	plan, err := newParallelResidualPlan(blocks, parallelResidual)
	if err != nil {
		return nil, err
	}
	if plan.any && unet {
		return nil, fmt.Errorf("parallel_residual is not supported with unet")
	}
	if len(executionOrder) > 0 && unet {
		return nil, fmt.Errorf("recurrence activation schedule is not supported with unet")
	}
	backoutPlan, err := newBackoutBuildPlan(backoutSpec, len(blocks), unet, "IR build")
	if err != nil {
		return nil, err
	}

	if useTiedHead && !reserveHead && !tieEmbeddings {
		return nil, fmt.Errorf("tied LM head requires embedding weight layout")
	}
	if !useTiedHead && !reserveHead {
		return nil, fmt.Errorf("untied LM head requires reserved head weight")
	}
	if mlmHead == MLMHeadBERT && !tieEmbeddings {
		return nil, fmt.Errorf("mlm_head=\"bert\" requires tie_embeddings=true")
	}

	nWeights, err := countWeightsWithFeaturesRecurrenceParallelHeadLayoutNorm(D, V, T, mlpMult, reserveHead, blockScales, residMix, unet, parallelResidual, charVocabSize, charDim, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, recurrence, norm, normPlacement, ffnInternalNorm)
	if err != nil {
		return nil, err
	}
	smearWeights, err := smearEmbeddingWeightShapes(D, T, smearOpts)
	if err != nil {
		return nil, err
	}
	nWeights += len(smearWeights)
	nWeights += len(backoutWeightShapes(backoutSpec))
	nWeights += data2VecWeightCount(data2vec)
	nWeights += mlmHeadWeightCount(D, V, mlmHead)

	prog := NewProgram(nWeights)
	maskedObjective := isMaskedTrainingObjective(objective)
	distillationEnabled := distillation != nil
	prog.DeclareInput("tokens", TensorInt32, []int{B, T})
	prog.DeclareInput("targets", TensorInt32, []int{B * T})
	if maskedObjective {
		prog.DeclareInput("loss_mask", TensorFloat32, []int{B * T})
	}
	if objective == ObjectiveHybridExample {
		prog.DeclareInput("attention_causal_mask", TensorInt32, []int{B})
	}
	if distillationEnabled {
		prog.DeclareInput("teacher_probs", TensorFloat32, []int{B * T, V})
	}
	data2VecEnabled := data2vec != nil && data2vec.LossWeight > 0
	if data2VecEnabled {
		prog.DeclareInput("data2vec_targets", TensorFloat32, []int{B * T, D})
		prog.DeclareInput("data2vec_loss_mask", TensorFloat32, []int{B * T})
	}
	if firstByteMask {
		prog.DeclareInput("first_byte_valid", TensorInt32, []int{V})
	}

	// Embedding lookup
	wi := 0
	prog.Embed(weightName(wi), "tokens", "x_embed")
	wi = fixedWeightCountWithHeadAndNorm(reserveHead, norm)
	embedState := "x_embed"
	if smearOpts.Enabled {
		embedState, wi, err = emitSmearEmbeddingIR(prog, embedState, T, D, wi, smearOpts)
		if err != nil {
			return nil, err
		}
	}

	// Flatten to [B*T, D], leaving room to inject optional model-level feature embeddings.
	xState := "x"
	if charVocabSize > 0 || bigramVocabSize > 0 || trigramVocabSize > 0 {
		xState = "x_tok"
	}
	prog.Reshape(embedState, []int{B * T, D}, xState)
	featureBase := xState
	wi = emitCharIR(prog, featureBase, B, T, D, wi, charVocabSize, charDim, charMaxPerToken)
	if charVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitBigramIR(prog, featureBase, B, T, D, wi, bigramVocabSize, bigramDim)
	if bigramVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitTrigramIR(prog, featureBase, B, T, D, wi, trigramVocabSize, trigramDim)
	sharedRel, err := newSharedRelativeAttentionPlan(blocks)
	if err != nil {
		return nil, err
	}
	if sharedRel.Enabled {
		sharedRel.WeightIndex = wi
		wi++
		sharedRel.NormEps = norm.Eps
		if sharedRel.Norm == RelativeAttentionEmbeddingNormLayerNorm {
			sharedRel.NormIndex = wi
			wi += 2
		}
	}
	if residMix {
		prog.ScalarMul("x", 1.0, "x0")
	}

	opIdx := 0

	// Sequential blocks process "x" directly.
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return nil, err
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return nil, err
	}
	weightStarts := make([]int, len(blocks))
	for i := range weightStarts {
		weightStarts[i] = -1
	}
	kvCache := make(map[int]BlockKVOutputs, len(blocks))
	switch {
	case unet:
		numEncoder, numSkip := unetLayout(len(blocks))
		for i := range blocks[:numEncoder] {
			wi, err = emitSequentialBlockWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, i, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, sharedRel)
			if err != nil {
				return nil, err
			}
			backoutPlan.captureAfterBlock(prog, i, "x")
			prog.ScalarMul("x", 1.0, fmt.Sprintf("enc_%d", i))
			opIdx++
		}
		skipBase := wi
		wi += numSkip
		for decIdx := range blocks[numEncoder:] {
			if numSkip > 0 {
				skipIdx := decIdx
				if skipIdx >= numSkip {
					skipIdx = numSkip - 1
				}
				encIdx := numEncoder - 1 - decIdx
				if encIdx < 0 {
					encIdx = 0
				}
				if encIdx >= numSkip {
					encIdx = numSkip - 1
				}
				skipScaled := fmt.Sprintf("unet_skip_%d", decIdx)
				prog.Mul(fmt.Sprintf("enc_%d", encIdx), weightName(skipBase+skipIdx), skipScaled)
				prog.Add("x", skipScaled, "x")
			}
			blockIdx := numEncoder + decIdx
			wi, err = emitSequentialBlockWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, blockIdx, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, sharedRel)
			if err != nil {
				return nil, err
			}
			backoutPlan.captureAfterBlock(prog, blockIdx, "x")
			opIdx++
		}
	case len(executionOrder) > 0:
		wi, err = emitSequentialOrderWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, executionOrder, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, parallelResidual, dropout, attnDropout, &backoutPlan, norm, normPlacement, ffnInternalNorm, sharedRel)
		if err != nil {
			return nil, err
		}
	default:
		if hiddenCapture != nil {
			for i := 0; i < len(blocks); i++ {
				wi, err = emitSequentialBlockWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, i, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, dropout, attnDropout, norm, normPlacement, ffnInternalNorm, sharedRel)
				if err != nil {
					return nil, err
				}
				backoutPlan.captureAfterBlock(prog, i, "x")
				hiddenCapture.captureAfterBlock(prog, i, "x")
				opIdx++
			}
			break
		}
		wi, err = emitSequentialRangeWithRecurrenceDropout(prog, blocks, refs, weightStarts, kvCache, 0, len(blocks), "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, parallelResidual, dropout, attnDropout, &backoutPlan, norm, normPlacement, ffnInternalNorm, sharedRel)
		if err != nil {
			return nil, err
		}
	}
	backoutPlan.setWeightIndex(wi)
	if backoutPlan.enabled {
		wi++
	}
	if err := backoutPlan.applyBeforeFinalNorm(prog, "x"); err != nil {
		return nil, err
	}

	// Final normalization
	if _, err := emitNamedNormIR(prog, "x", finalNormWeightIndexWithHeadAndNorm(reserveHead, norm), "x_final_norm", norm); err != nil {
		return nil, err
	}
	prog.Reshape("x_final_norm", []int{B, T, D}, "x_hidden")
	if data2VecEnabled {
		wi = emitData2VecPredictorIR(prog, data2vec, wi)
	}

	// Output head projection
	logitsState := "logits"
	if maskedObjective && mlmHead == MLMHeadBERT {
		var err error
		logitsState, wi, err = emitBERTMLMHeadIR(prog, "x_final_norm", wi, D, V, norm.Eps, dropout)
		if err != nil {
			return nil, err
		}
	} else {
		if useTiedHead {
			prog.Transpose(weightName(0), []int{1, 0}, "tied_head")
			prog.MatMul("x_final_norm", "tied_head", "logits")
		} else {
			prog.MatMul("x_final_norm", weightName(1), "logits")
		}
		if mlmHead == MLMHeadBERT {
			wi += mlmHeadWeightCount(D, V, mlmHead)
		}
	}

	if logitSoftcap > 0 {
		prog.ScalarMul(logitsState, 1.0/logitSoftcap, "logits_softcap_scaled")
		prog.Tanh("logits_softcap_scaled", "logits_softcap_tanh")
		prog.ScalarMul("logits_softcap_tanh", logitSoftcap, "logits_softcapped")
		logitsState = "logits_softcapped"
	}
	if logitsState != "logits" {
		prog.ScalarMul(logitsState, 1.0, "logits")
	}

	switch {
	case maskedObjective:
		emitMaskedLanguageModelLossIR(prog, logitsState, "targets", "loss_mask")
	case distillationEnabled:
		if err := emitDistillationLanguageModelLossIR(prog, logitsState, "targets", "teacher_probs", distillation.LossWeightCE, distillation.LossWeightKL); err != nil {
			return nil, err
		}
	default:
		if err := emitLanguageModelLossIR(prog, logitsState, "targets", B, T, V, mtp, firstByteMask); err != nil {
			return nil, err
		}
	}
	if data2VecEnabled {
		emitData2VecLossIR(prog, data2vec)
	}
	taskLossHasEvalLoss := maskedObjective || distillationEnabled || firstByteMask || (mtp != nil && mtp.EffectiveN() > 1)
	taskLossHasEvalLoss = emitZLossIR(prog, logitsState, zLoss, taskLossHasEvalLoss)
	moeEnabled := emitMoEAuxiliaryAggregatesIR(prog, taskLossHasEvalLoss)
	prog.DeclareOutput("loss", TensorFloat32, []int{1})
	if taskLossHasEvalLoss || moeEnabled {
		prog.DeclareOutput("eval_loss", TensorFloat32, []int{1})
	}
	if moeEnabled {
		prog.DeclareOutput("moe_aux_loss", TensorFloat32, []int{1})
		prog.DeclareOutput("moe_router_entropy", TensorFloat32, []int{1})
	}
	prog.DeclareOutput("per_token_nll", TensorFloat32, []int{B * T})
	prog.DeclareOutput("x_hidden", TensorFloat32, []int{B, T, D})
	prog.DeclareOutput("logits", TensorFloat32, []int{B * T, V})
	if data2VecEnabled {
		prog.DeclareOutput("data2vec_loss", TensorFloat32, []int{1})
	}
	hiddenCapture.declareOutputs(prog, B*T, D)

	// Verify weight count consistency
	if wi != nWeights {
		return nil, fmt.Errorf("IR weight count mismatch: emitted=%d expected=%d", wi, nWeights)
	}

	return prog, nil
}
