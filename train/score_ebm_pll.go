package train

import (
	"fmt"
	"sort"
	"strings"
)

type scoreEBMPLLAttributionRecord struct {
	ID                   string     `json:"id"`
	Family               string     `json:"family,omitempty"`
	CleanTokens          []int      `json:"clean_tokens"`
	CorruptTokens        []int      `json:"corrupt_tokens"`
	CleanLogprobs        []*float64 `json:"clean_logprobs"`
	CorruptLogprobs      []*float64 `json:"corrupt_logprobs"`
	DifferingSpanClean   []int      `json:"differing_span_clean"`
	DifferingSpanCorrupt []int      `json:"differing_span_corrupt"`
	ScoreClean           float64    `json:"score_clean"`
	ScoreCorrupt         float64    `json:"score_corrupt"`
	Margin               float64    `json:"margin"`
	Correct              bool       `json:"correct"`
	Aggregation          string     `json:"aggregation"`
	SkippedTokenIDs      []int      `json:"skipped_token_ids"`
}

func scoreEBMPLLAggregationFromConfig(raw string) bool {
	mode := strings.ToLower(strings.TrimSpace(raw))
	return mode == "" || mode == scoreEBMPLLAggregationConfig
}

func scoreEBMUsesPositionPLL(scoreMode, pllAggregation string, fromConfig bool) bool {
	if scoreMode == scoreEBMModeEnergy {
		return false
	}
	switch pllAggregation {
	case scoreEBMPLLAggregationFullSeq, scoreEBMPLLAggregationDependentWin:
		return true
	case scoreEBMPLLAggregationDifferingSpan:
		return !fromConfig
	default:
		return false
	}
}

func validateScoreEBMPLLWindow(pllAggregation string, window int) error {
	if window < 0 {
		return fmt.Errorf("-score-pll-window must be >= 0")
	}
	if window > 0 && pllAggregation != scoreEBMPLLAggregationDependentWin {
		return fmt.Errorf("-score-pll-window is only valid with -score-pll-aggregation=%s", scoreEBMPLLAggregationDependentWin)
	}
	return nil
}

func scoreEBMPLLRecordWithOptions(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, rec scoreEBMInputRecord, opts scoreEBMRuntimeOptions, aggregation string) (scoreEBMOutputRecord, error) {
	if err := validateScoreEBMPLLWindow(aggregation, opts.pllWindow); err != nil {
		return scoreEBMOutputRecord{}, err
	}
	positionBatch := opts.scorePositionBatch
	if positionBatch <= 0 {
		var err error
		positionBatch, err = effectiveScorePositionBatch(cfg.SeqLen, cfg.VocabSize, positionBatch)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
	}
	skipIDs := scoreEBMEffectivePLLSkipTokenIDs(cfg, opts.pllSkipTokenIDs)
	if len(rec.Tokens) > 0 {
		score, err := scoreEBMPLLScoreOne(cfg, evaluator, rec.Tokens, rec.Span, aggregation, opts.pllWindow, positionBatch, skipIDs, false)
		if err != nil {
			return scoreEBMOutputRecord{}, err
		}
		v := float64(float32(score))
		return scoreEBMOutputRecord{ID: rec.ID, NTokens: len(rec.Tokens), Score: &v}, nil
	}
	pair := minimalPairRecord{
		ID:          rec.ID,
		Clean:       rec.Clean,
		Corrupt:     rec.Corrupt,
		CleanSpan:   append([]int(nil), rec.CleanSpan...),
		CorruptSpan: append([]int(nil), rec.CorruptSpan...),
		Family:      rec.Family,
	}
	if aggregation != scoreEBMPLLAggregationFullSeq || opts.pllAttributionEnc != nil {
		if err := ensureMinimalPairRecordSpans(&pair); err != nil {
			return scoreEBMOutputRecord{}, err
		}
	}
	cleanSpan := pair.CleanSpan
	corruptSpan := pair.CorruptSpan
	clean, cleanTrace, err := scoreEBMPLLScorePairSide(cfg, evaluator, rec.Clean, cleanSpan, aggregation, opts.pllWindow, positionBatch, skipIDs, opts.pllAttributionEnc != nil)
	if err != nil {
		return scoreEBMOutputRecord{}, fmt.Errorf("clean: %w", err)
	}
	corrupt, corruptTrace, err := scoreEBMPLLScorePairSide(cfg, evaluator, rec.Corrupt, corruptSpan, aggregation, opts.pllWindow, positionBatch, skipIDs, opts.pllAttributionEnc != nil)
	if err != nil {
		return scoreEBMOutputRecord{}, fmt.Errorf("corrupt: %w", err)
	}
	cleanOut := float64(float32(clean))
	corruptOut := float64(float32(corrupt))
	margin := cleanOut - corruptOut
	correct := cleanOut > corruptOut
	out := scoreEBMOutputRecord{
		ID:           rec.ID,
		Family:       rec.Family,
		ScoreClean:   &cleanOut,
		ScoreCorrupt: &corruptOut,
		Margin:       &margin,
		Correct:      &correct,
	}
	if opts.pllAttributionEnc != nil {
		attr := scoreEBMPLLAttributionRecord{
			ID:                   rec.ID,
			Family:               rec.Family,
			CleanTokens:          append([]int(nil), rec.Clean...),
			CorruptTokens:        append([]int(nil), rec.Corrupt...),
			CleanLogprobs:        cleanTrace.Logprobs,
			CorruptLogprobs:      corruptTrace.Logprobs,
			DifferingSpanClean:   append([]int(nil), cleanSpan...),
			DifferingSpanCorrupt: append([]int(nil), corruptSpan...),
			ScoreClean:           cleanOut,
			ScoreCorrupt:         corruptOut,
			Margin:               margin,
			Correct:              correct,
			Aggregation:          aggregation,
			SkippedTokenIDs:      sortedScoreEBMSkipIDs(skipIDs),
		}
		if err := opts.pllAttributionEnc.Encode(attr); err != nil {
			return scoreEBMOutputRecord{}, fmt.Errorf("write PLL attribution for %q: %w", rec.ID, err)
		}
	}
	return out, nil
}

func scoreEBMPLLScoreOne(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, tokens, span []int, aggregation string, window, positionBatch int, skipIDs map[int]bool, attribution bool) (float64, error) {
	if err := validateScoreDiffusionTokens(cfg, tokens, 0); err != nil {
		return 0, err
	}
	positions, err := scoreEBMPLLAggregatePositions(tokens, span, aggregation, window, skipIDs, true)
	if err != nil {
		return 0, err
	}
	tracePositions := positions
	if attribution {
		tracePositions = scoreEBMFullSeqPLLPositions(tokens, skipIDs)
	}
	trace, err := scoreEBMPLLTraceSequence(cfg, evaluator, tokens, tracePositions, positionBatch)
	if err != nil {
		return 0, err
	}
	return trace.sumPositions(positions), nil
}

func scoreEBMPLLScorePairSide(cfg *ArchConfig, evaluator diffusionGenerationEvaluator, tokens, span []int, aggregation string, window, positionBatch int, skipIDs map[int]bool, attribution bool) (float64, scoreEBMPLLTrace, error) {
	if err := validateScoreDiffusionTokens(cfg, tokens, 0); err != nil {
		return 0, scoreEBMPLLTrace{}, err
	}
	positions, err := scoreEBMPLLAggregatePositions(tokens, span, aggregation, window, skipIDs, false)
	if err != nil {
		return 0, scoreEBMPLLTrace{}, err
	}
	tracePositions := positions
	if attribution {
		tracePositions = scoreEBMFullSeqPLLPositions(tokens, skipIDs)
	}
	trace, err := scoreEBMPLLTraceSequence(cfg, evaluator, tokens, tracePositions, positionBatch)
	if err != nil {
		return 0, scoreEBMPLLTrace{}, err
	}
	return trace.sumPositions(positions), trace, nil
}

func scoreEBMPLLAggregatePositions(tokens, span []int, aggregation string, window int, skipIDs map[int]bool, requireExplicitSpan bool) ([]int, error) {
	switch aggregation {
	case scoreEBMPLLAggregationFullSeq:
		return scoreEBMFullSeqPLLPositions(tokens, skipIDs), nil
	case scoreEBMPLLAggregationDifferingSpan, scoreEBMPLLAggregationDependentWin:
		if requireExplicitSpan && len(span) == 0 {
			return nil, fmt.Errorf("-score-pll-aggregation=%s requires span for single-sequence records", aggregation)
		}
		start, end, ok, err := parseMinimalPairSpan("span", span, len(tokens))
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("internal error: missing PLL span")
		}
		if aggregation == scoreEBMPLLAggregationDependentWin {
			start -= window
			if start < 0 {
				start = 0
			}
			end += window
			if end > len(tokens) {
				end = len(tokens)
			}
		}
		return scoreEBMPLLPositionsInRange(tokens, start, end, skipIDs), nil
	default:
		return nil, fmt.Errorf("-score-pll-aggregation=%q is not supported for PLL scoring", aggregation)
	}
}

func scoreEBMPLLPositionsInRange(tokens []int, start, end int, skipIDs map[int]bool) []int {
	positions := make([]int, 0, end-start)
	for pos := start; pos < end; pos++ {
		if skipIDs != nil && skipIDs[tokens[pos]] {
			continue
		}
		positions = append(positions, pos)
	}
	return positions
}

func scoreEBMEffectivePLLSkipTokenIDs(cfg *ArchConfig, skipTokenIDs map[int]bool) map[int]bool {
	out := make(map[int]bool, len(skipTokenIDs)+1)
	for id := range skipTokenIDs {
		out[id] = true
	}
	if cfg != nil && cfg.Training.MLMMaskTokenID >= 0 && cfg.Training.MLMMaskTokenID < cfg.VocabSize {
		out[cfg.Training.MLMMaskTokenID] = true
	}
	return out
}

func sortedScoreEBMSkipIDs(skipIDs map[int]bool) []int {
	out := make([]int, 0, len(skipIDs))
	for id := range skipIDs {
		out = append(out, id)
	}
	sort.Ints(out)
	return out
}
