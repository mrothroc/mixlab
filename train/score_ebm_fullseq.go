package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
)

// Full-sequence pseudo-log-likelihood scoring for score-ebm: mask each scorable
// position (in position-batched chunks) and sum the scorer head log-probs.

func scoreEBMFullSeqPLLSequences(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, sequences [][]int, opts scoreEBMRuntimeOptions) ([]float32, [][]float32, error) {
	positionBatch := opts.scorePositionBatch
	if positionBatch <= 0 {
		var err error
		positionBatch, err = effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, positionBatch)
		if err != nil {
			return nil, nil, err
		}
	}
	out := make([]float32, len(sequences))
	for i, tokens := range sequences {
		score, err := scoreEBMFullSeqPLLSequence(cfg, evaluator, tokens, positionBatch, opts.pllSkipTokenIDs)
		if err != nil {
			return nil, nil, fmt.Errorf("sequence %d: %w", i, err)
		}
		out[i] = float32(score)
	}
	return out, nil, nil
}

func scoreEBMFullSeqPLLSequence(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, tokens []int, positionBatch int, skipTokenIDs map[int]bool) (float64, error) {
	if cfg == nil {
		return 0, fmt.Errorf("nil config")
	}
	if evaluator == nil {
		return 0, fmt.Errorf("nil EBM evaluator")
	}
	if positionBatch <= 0 {
		return 0, fmt.Errorf("invalid score position batch=%d", positionBatch)
	}
	effectiveSkip := make(map[int]bool, len(skipTokenIDs)+1)
	for id := range skipTokenIDs {
		effectiveSkip[id] = true
	}
	if cfg.Training.MLMMaskTokenID >= 0 && cfg.Training.MLMMaskTokenID < cfg.VocabSize {
		effectiveSkip[cfg.Training.MLMMaskTokenID] = true
	}
	positions := scoreEBMFullSeqPLLPositions(tokens, effectiveSkip)
	if len(positions) == 0 {
		return 0, nil
	}
	mode, err := scoreEBMMode(cfg)
	if err != nil {
		return 0, err
	}
	outputName := "logits"
	if mode == scoreEBMModeMLMSpanPLL {
		outputName, err = mlmSpanPLLScoreLogitsOutputName(cfg)
		if err != nil {
			return 0, err
		}
	} else if mode != scoreEBMModeSinglePLL {
		return 0, fmt.Errorf("full-sequence PLL scoring requires an MLM/MNTP scorer config")
	}
	var total float64
	for start := 0; start < len(positions); start += positionBatch {
		end := start + positionBatch
		if end > len(positions) {
			end = len(positions)
		}
		chunk := positions[start:end]
		batch, err := fullSeqPLLScoringBatch(cfg, tokens, chunk, positionBatch, mode)
		if err != nil {
			return 0, err
		}
		evalBatchSize := positionBatch
		if batch.batchSizeOverride > 0 {
			evalBatchSize = batch.batchSizeOverride
		}
		if _, err := evaluateObjectiveAndCacheOutputs(evaluator, batch, evalBatchSize, cfg.SeqLen, outputName); err != nil {
			return 0, err
		}
		logits, err := evaluator.ReadOutput(outputName, []int{positionBatch * cfg.SeqLen, cfg.VocabSize})
		if err != nil {
			return 0, fmt.Errorf("read %s: %w", outputName, err)
		}
		want := positionBatch * cfg.SeqLen * cfg.VocabSize
		if len(logits) != want {
			return 0, fmt.Errorf("PLL logits length mismatch: got=%d want=%d", len(logits), want)
		}
		for row, pos := range chunk {
			logitStart := (row*cfg.SeqLen + pos) * cfg.VocabSize
			lp, err := targetLogProbFromLogits(logits[logitStart:logitStart+cfg.VocabSize], tokens[pos])
			if err != nil {
				return 0, fmt.Errorf("position %d: %w", pos, err)
			}
			total += lp
		}
	}
	return total, nil
}

func scoreEBMFullSeqPLLPositions(tokens []int, skipTokenIDs map[int]bool) []int {
	positions := make([]int, 0, len(tokens))
	for pos, token := range tokens {
		if skipTokenIDs != nil && skipTokenIDs[token] {
			continue
		}
		positions = append(positions, pos)
	}
	return positions
}

func fullSeqPLLScoringBatch(cfg *ArchConfig, tokens []int, positions []int, batchSize int, mode string) (objectiveBatch, error) {
	if cfg == nil {
		return objectiveBatch{}, fmt.Errorf("nil config")
	}
	if len(positions) == 0 {
		return objectiveBatch{}, fmt.Errorf("positions must be non-empty")
	}
	if batchSize <= 0 {
		return objectiveBatch{}, fmt.Errorf("invalid batch size=%d", batchSize)
	}
	if len(positions) > batchSize {
		return objectiveBatch{}, fmt.Errorf("positions=%d exceeds batch size=%d", len(positions), batchSize)
	}
	if len(tokens) > cfg.SeqLen {
		return objectiveBatch{}, fmt.Errorf("token length %d exceeds seq_len %d", len(tokens), cfg.SeqLen)
	}
	for _, pos := range positions {
		if pos < 0 || pos >= len(tokens) {
			return objectiveBatch{}, fmt.Errorf("position %d outside token length %d", pos, len(tokens))
		}
	}
	if mode == scoreEBMModeMLMSpanPLL {
		return fullSeqPLLMultiheadScoringBatch(cfg, tokens, positions, batchSize)
	}
	return fullSeqPLLSingleScoringBatch(cfg, tokens, positions, batchSize)
}

func fullSeqPLLSingleScoringBatch(cfg *ArchConfig, tokens []int, positions []int, batchSize int) (objectiveBatch, error) {
	need := batchSize * cfg.SeqLen
	x := make([]int, need)
	y := make([]int, need)
	lossMask := make([]float32, need)
	unmasked := make([]int, need)
	fillMaskedPLLRows(cfg, x, y, unmasked, lossMask, tokens, positions, batchSize, cfg.Training.MLMMaskTokenID)
	return objectiveBatch{x: x, y: y, lossMask: lossMask, unmaskedX: unmasked}, nil
}

func fullSeqPLLMultiheadScoringBatch(cfg *ArchConfig, tokens []int, positions []int, rawBatchSize int) (objectiveBatch, error) {
	headCount := len(cfg.Training.Heads)
	need := rawBatchSize * cfg.SeqLen
	totalRows := rawBatchSize * headCount
	if minimalPairUsesMLMSpanPLL(cfg) {
		totalRows += rawBatchSize
	}
	totalNeed := totalRows * cfg.SeqLen
	x := make([]int, totalNeed)
	y := make([]int, totalNeed)
	lossMask := make([]float32, totalNeed)
	energySpanMask := make([]float32, totalNeed)
	starts := make([]int32, totalRows)
	ends := make([]int32, totalRows)
	timestep := make([]float32, totalRows)
	pad := minimalPairPadTokenID(cfg)
	scoreHeadName, err := mlmSpanPLLScoreHeadName(cfg)
	if err != nil {
		return objectiveBatch{}, err
	}
	for headIdx, head := range cfg.Training.Heads {
		tokenOffset := headIdx * need
		rowOffset := headIdx * rawBatchSize
		for row := 0; row < rawBatchSize; row++ {
			rowStart := tokenOffset + row*cfg.SeqLen
			fillMinimalPairRow(x[rowStart:rowStart+cfg.SeqLen], y[rowStart:rowStart+cfg.SeqLen], tokens, pad)
			if head.Name == scoreHeadName {
				pos := positions[0]
				if row < len(positions) {
					pos = positions[row]
					lossMask[rowStart+pos] = 1
				}
				if pos < 0 || pos >= len(tokens) {
					return objectiveBatch{}, fmt.Errorf("position %d outside token length %d", pos, len(tokens))
				}
				x[rowStart+pos] = cfg.Training.MLMMaskTokenID
			}
			starts[rowOffset+row] = 0
			if head.Objective == arch.ObjectiveCausal {
				ends[rowOffset+row] = 0
			} else {
				ends[rowOffset+row] = int32(cfg.SeqLen)
			}
		}
	}
	if minimalPairUsesMLMSpanPLL(cfg) {
		pairOffset := headCount * need
		pairRowOffset := headCount * rawBatchSize
		for row := 0; row < rawBatchSize; row++ {
			rowStart := pairOffset + row*cfg.SeqLen
			fillMinimalPairRow(x[rowStart:rowStart+cfg.SeqLen], y[rowStart:rowStart+cfg.SeqLen], tokens, pad)
			starts[pairRowOffset+row] = 0
			ends[pairRowOffset+row] = int32(cfg.SeqLen)
		}
	}
	return objectiveBatch{
		x:                   x,
		y:                   y,
		lossMask:            lossMask,
		energySpanMask:      energySpanMask,
		diffusionBlockStart: starts,
		diffusionBlockEnd:   ends,
		diffusionTimestep:   timestep,
		batchSizeOverride:   totalRows,
	}, nil
}

func fillMaskedPLLRows(cfg *ArchConfig, x, y, unmasked []int, lossMask []float32, tokens []int, positions []int, batchSize, maskTokenID int) {
	for row := 0; row < batchSize; row++ {
		pos := positions[0]
		active := row < len(positions)
		if active {
			pos = positions[row]
		}
		rowStart := row * cfg.SeqLen
		copy(x[rowStart:rowStart+len(tokens)], tokens)
		copy(y[rowStart:rowStart+len(tokens)], tokens)
		copy(unmasked[rowStart:rowStart+len(tokens)], tokens)
		x[rowStart+pos] = maskTokenID
		if active {
			lossMask[rowStart+pos] = 1
		}
	}
}
