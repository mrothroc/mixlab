package arch

import "fmt"

// Weight-counting helpers for the n-gram/recurrence/parallel-residual weight
// layout. These mirror the shapes produced by weight_shapes.go.

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
