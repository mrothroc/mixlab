package arch

// Public CountWeights* helpers: thin delegations exposing IR weight-tensor
// counts to downstream packages. The real layout logic lives in builder.go.

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
	return countWeightsWithBigramRecurrenceParallelHeadLayout(modelDim, mlpMult, !tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, blocks, recurrence)
}
