package arch

import "fmt"

// CollectWeightShapes returns the complete ordered list of WeightMeta for an
// architecture configuration. The order matches the weight indices used by
// BuildIRProgram: embed, head, final_norm, then block weights in emission order.
//
// This is the single source of truth for weight shapes. Both IR building and
// weight initialization should derive from this function.
func CollectWeightShapes(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	blocks []BlockSpec,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigram(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, 0, 0, blocks)
}

// CollectWeightShapesWithBigram returns ordered weight metadata including
// optional model-level bigram embedding weights.
func CollectWeightShapesWithBigram(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigramAndRecurrence(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, bigramVocabSize, bigramDim, blocks, nil)
}

// CollectWeightShapesWithBigramAndRecurrence returns ordered weight metadata
// including optional bigram weights and only original sequential block weights.
func CollectWeightShapesWithBigramAndRecurrence(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigramRecurrenceAndParallel(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, false, bigramVocabSize, bigramDim, blocks, recurrence)
}

// CollectWeightShapesWithBigramRecurrenceAndParallel returns ordered weight
// metadata including optional bigram weights, original sequential block
// weights, and parallel residual norm sharing.
func CollectWeightShapesWithBigramRecurrenceAndParallel(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	return collectWeightShapesWithRefs(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, 0, 0, blocks, refs)
}

// CollectWeightShapesWithNgramsRecurrenceAndParallel returns ordered weight
// metadata including optional bigram and trigram weights, original sequential
// block weights, and parallel residual norm sharing.
func CollectWeightShapesWithNgramsRecurrenceAndParallel(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	return collectWeightShapesWithRefs(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, refs)
}
