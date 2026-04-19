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
	return strings.ToLower(strings.TrimSpace(spec.Type))
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

func countStreamWeightsWithRecurrence(specs []BlockSpec, recurrence []int, blockScales, residMix bool) (int, error) {
	rec, err := normalizeRecurrence(specs, recurrence)
	if err != nil {
		return 0, err
	}
	total := 0
	for i, spec := range specs {
		if rec[i] != i {
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

func countBlockRangeWeightsWithRecurrence(specs []BlockSpec, rec []int, start, end int, blockScales, residMix bool) (int, error) {
	total := 0
	for i := start; i < end; i++ {
		if rec[i] != i {
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

func bigramWeightCount(modelDim, bigramVocabSize, bigramDim int) int {
	if bigramVocabSize <= 0 {
		return 0
	}
	if bigramDim <= 0 {
		bigramDim = modelDim
	}
	count := 2 // embed table + learned scale
	if bigramDim != modelDim {
		count++
	}
	return count
}

func emitBigramIR(prog *Program, B, T, D, wi, bigramVocabSize, bigramDim int) int {
	if bigramVocabSize <= 0 {
		return wi
	}
	if bigramDim <= 0 {
		bigramDim = D
	}
	prog.DeclareInput("bigram_ids", TensorInt32, []int{B, T})
	prog.Embed(weightName(wi), "bigram_ids", "bigram_embed")
	wi++
	prog.Reshape("bigram_embed", []int{B * T, bigramDim}, "bigram_flat")
	bigramState := "bigram_flat"
	if bigramDim != D {
		prog.MatMul(bigramState, weightName(wi), "bigram_proj")
		wi++
		bigramState = "bigram_proj"
	}
	prog.Mul(bigramState, weightName(wi), "bigram_scaled")
	wi++
	prog.Add("x_tok", "bigram_scaled", "x")
	return wi
}

// CountWeights returns the total number of IR weight tensors for a given
// architecture configuration. This is used to validate weight registration.
//
// Architecture layout:
//
//	w0 = embedding table
//	w1 = output head projection
//	w2 = final RMSNorm scale
//	w3... = block weights
func CountWeights(
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	blocks []BlockSpec,
) (int, error) {
	return CountWeightsWithBigram(0, mlpMult, tieEmbeddings, blockScales, residMix, unet, 0, 0, blocks)
}

// CountWeightsWithBigram returns the IR weight tensor count, including
// optional model-level bigram embedding weights.
func CountWeightsWithBigram(
	modelDim int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
) (int, error) {
	return CountWeightsWithBigramAndRecurrence(modelDim, mlpMult, tieEmbeddings, blockScales, residMix, unet, bigramVocabSize, bigramDim, blocks, nil)
}

// CountWeightsWithBigramAndRecurrence returns the IR weight tensor count,
// including optional bigram embeddings and sequential block weight tying.
func CountWeightsWithBigramAndRecurrence(
	modelDim int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) (int, error) {
	return CountWeightsWithBigramRecurrenceAndParallel(modelDim, mlpMult, tieEmbeddings, blockScales, residMix, unet, false, bigramVocabSize, bigramDim, blocks, recurrence)
}

// CountWeightsWithBigramRecurrenceAndParallel returns the IR weight tensor
// count, including optional bigram embeddings, sequential block weight tying,
// and parallel residual block pairs.
func CountWeightsWithBigramRecurrenceAndParallel(
	modelDim int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) (int, error) {
	_ = mlpMult
	total := fixedWeightCount(tieEmbeddings)
	total += bigramWeightCount(modelDim, bigramVocabSize, bigramDim)

	if parallelResidual {
		if err := validateParallelResidualBlocks(blocks); err != nil {
			return 0, err
		}
		if unet {
			return 0, fmt.Errorf("parallel_residual is not supported with unet")
		}
	}
	rec, err := normalizeRecurrence(blocks, recurrence)
	if err != nil {
		return 0, fmt.Errorf("blocks: %w", err)
	}
	if unet {
		numEncoder, numSkip := unetLayout(len(blocks))
		n, err := countBlockRangeWeightsWithRecurrenceAndParallel(blocks, rec, 0, numEncoder, blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n + numSkip
		n, err = countBlockRangeWeightsWithRecurrenceAndParallel(blocks, rec, numEncoder, len(blocks), blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n
	} else {
		n, err := countStreamWeightsWithRecurrenceAndParallel(blocks, recurrence, blockScales, residMix, parallelResidual)
		if err != nil {
			return 0, fmt.Errorf("blocks: %w", err)
		}
		total += n
	}

	return total, nil
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

func fixedWeightCount(tieEmbeddings bool) int {
	if tieEmbeddings {
		return 2 // embed + final_norm
	}
	return 3 // embed + head + final_norm
}

func finalNormWeightIndex(tieEmbeddings bool) int {
	if tieEmbeddings {
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
	return emitStreamIRWithDropout(prog, specs, stream, original, wi, D, T, B, V, opIdx, mlpMult, blockScales, residMix, 0)
}

func emitStreamIRWithDropout(prog *Program, specs []BlockSpec, stream, original string, wi, D, T, B, V int, opIdx *int, mlpMult float64, blockScales, residMix bool, dropout float32) (int, error) {
	for _, spec := range specs {
		var err error
		if needsResidMix(spec, residMix) {
			wi = applyResidMixIR(prog, stream, original, wi, D, *opIdx)
		}
		wi, err = emitBlockIRWithDropout(prog, spec, stream, wi, D, T, B, V, *opIdx, nil, mlpMult, blockScales, dropout)
		if err != nil {
			return wi, err
		}
		(*opIdx)++
	}
	return wi, nil
}


func emitSequentialBlockWithRecurrenceDropout(prog *Program, specs []BlockSpec, rec []int, weightStarts []int, blockIdx int, stream, original string, wi, D, T, B, V int, opIdx *int, streamSeqLens map[string]int, mlpMult float64, blockScales, residMix bool, dropout float32) (int, error) {
	spec := specs[blockIdx]
	blockWI := wi
	originalBlock := rec[blockIdx] == blockIdx
	if !originalBlock {
		blockWI = weightStarts[rec[blockIdx]]
		if blockWI < 0 {
			return wi, fmt.Errorf("recurrence[%d]=%d references block without emitted weights", blockIdx, rec[blockIdx])
		}
	}

	bodyWI := blockWI
	if needsResidMix(spec, residMix) {
		bodyWI = applyResidMixIR(prog, stream, original, bodyWI, D, *opIdx)
	}
	nextWI, err := emitBlockIRWithDropout(prog, spec, stream, bodyWI, D, T, B, V, *opIdx, streamSeqLens, mlpMult, blockScales, dropout)
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
	return emitBlockIRWithDropout(prog, spec, stream, wi, D, T, B, V, idx, streamSeqLens, mlpMult, blockScales, 0)
}

func emitBlockIRWithDropout(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, streamSeqLens map[string]int, mlpMult float64, blockScales bool, dropout float32) (int, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return wi, err
	}
	if reg.Emitter == nil {
		return wi, fmt.Errorf("block type %q has no emitter", spec.Type)
	}
	return reg.Emitter(prog, spec, stream, wi, D, T, B, V, idx, EmitOptions{
		StreamSeqLens: streamSeqLens,
		MLPMult:       mlpMult,
		BlockScales:   blockScales,
		Dropout:       dropout,
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
	if mlpMult <= 0 {
		mlpMult = DefaultFFNMultiplier
	}

	B := batchSize
	T := seqLen
	D := modelDim
	V := vocabSize

	if B <= 0 || T <= 0 {
		return nil, fmt.Errorf("invalid shape B=%d T=%d", B, T)
	}
	if D <= 0 {
		return nil, fmt.Errorf("invalid model_dim=%d", D)
	}
	if V <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", V)
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
	if parallelResidual {
		if err := validateParallelResidualBlocks(blocks); err != nil {
			return nil, err
		}
		if unet {
			return nil, fmt.Errorf("parallel_residual is not supported with unet")
		}
	}

	nWeights, err := CountWeightsWithBigramRecurrenceAndParallel(D, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, blocks, recurrence)
	if err != nil {
		return nil, err
	}

	prog := NewProgram(nWeights)
	prog.DeclareInput("tokens", TensorInt32, []int{B, T})
	prog.DeclareInput("targets", TensorInt32, []int{B * T})

	// Embedding lookup
	wi := 0
	prog.Embed(weightName(wi), "tokens", "x_embed")
	wi = fixedWeightCount(tieEmbeddings)

	// Flatten to [B*T, D], leaving room to inject model-level bigram embeddings.
	xState := "x"
	if bigramVocabSize > 0 {
		xState = "x_tok"
	}
	prog.Reshape("x_embed", []int{B * T, D}, xState)
	wi = emitBigramIR(prog, B, T, D, wi, bigramVocabSize, bigramDim)
	if residMix {
		prog.ScalarMul("x", 1.0, "x0")
	}

	opIdx := 0

	// Sequential blocks process "x" directly.
	rec, err := normalizeRecurrence(blocks, recurrence)
	if err != nil {
		return nil, err
	}
	weightStarts := make([]int, len(blocks))
	for i := range weightStarts {
		weightStarts[i] = -1
	}
	if unet {
		numEncoder, numSkip := unetLayout(len(blocks))
		for i := range blocks[:numEncoder] {
			wi, err = emitSequentialBlockWithRecurrenceDropout(prog, blocks, rec, weightStarts, i, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, dropout)
			if err != nil {
				return nil, err
			}
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
			wi, err = emitSequentialBlockWithRecurrenceDropout(prog, blocks, rec, weightStarts, blockIdx, "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, dropout)
			if err != nil {
				return nil, err
			}
			opIdx++
		}
	} else {
		wi, err = emitSequentialRangeWithRecurrenceDropout(prog, blocks, rec, weightStarts, 0, len(blocks), "x", "x0", wi, D, T, B, V, &opIdx, nil, mlpMult, blockScales, residMix, parallelResidual, dropout)
		if err != nil {
			return nil, err
		}
	}

	// Final normalization
	prog.RMSNorm("x", weightName(finalNormWeightIndex(tieEmbeddings)), "x_final_norm", 1e-5)
	prog.Reshape("x_final_norm", []int{B, T, D}, "x_hidden")

	// Output head projection
	if tieEmbeddings {
		prog.Transpose(weightName(0), []int{1, 0}, "tied_head")
		prog.MatMul("x_final_norm", "tied_head", "logits")
	} else {
		prog.MatMul("x_final_norm", weightName(1), "logits")
	}

	logitsState := "logits"
	if logitSoftcap > 0 {
		prog.ScalarMul(logitsState, 1.0/logitSoftcap, "logits_softcap_scaled")
		prog.Tanh("logits_softcap_scaled", "logits_softcap_tanh")
		prog.ScalarMul("logits_softcap_tanh", logitSoftcap, "logits_softcapped")
		logitsState = "logits_softcapped"
	}
	if logitsState != "logits" {
		prog.ScalarMul(logitsState, 1.0, "logits")
	}

	// Cross-entropy loss
	prog.CrossEntropy(logitsState, "targets", "loss")
	prog.DeclareOutput("loss", TensorFloat32, []int{1})
	prog.DeclareOutput("x_hidden", TensorFloat32, []int{B, T, D})
	prog.DeclareOutput("logits", TensorFloat32, []int{B * T, V})

	// Verify weight count consistency
	if wi != nWeights {
		return nil, fmt.Errorf("IR weight count mismatch: emitted=%d expected=%d", wi, nWeights)
	}

	return prog, nil
}
