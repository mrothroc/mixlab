package arch

import "fmt"

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
