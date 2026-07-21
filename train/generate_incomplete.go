package train

import (
	"fmt"
	"io"
)

type grammarIncompletePolicy string

const (
	grammarIncompleteError grammarIncompletePolicy = "error"
	grammarIncompleteSkip  grammarIncompletePolicy = "skip"

	defaultGrammarAttemptMultiplier = 4
)

type generationRunStats struct {
	Attempts          int
	Completed         int
	SkippedIncomplete int
}

func resolveGrammarIncompletePolicy(raw string, numSamples, configuredMaxAttempts int) (grammarIncompletePolicy, int, error) {
	policy := grammarIncompletePolicy(raw)
	if policy == "" {
		policy = grammarIncompleteError
	}
	switch policy {
	case grammarIncompleteError:
		if configuredMaxAttempts != 0 {
			return "", 0, fmt.Errorf("-grammar-max-attempts requires -grammar-on-incomplete=skip")
		}
		return policy, numSamples, nil
	case grammarIncompleteSkip:
		if configuredMaxAttempts < 0 {
			return "", 0, fmt.Errorf("-grammar-max-attempts must be >= 0")
		}
		maxAttempts := configuredMaxAttempts
		if maxAttempts == 0 {
			maxInt := int(^uint(0) >> 1)
			if numSamples > maxInt/defaultGrammarAttemptMultiplier {
				return "", 0, fmt.Errorf("default grammar attempt cap overflows for -num-samples=%d", numSamples)
			}
			maxAttempts = defaultGrammarAttemptMultiplier * numSamples
		}
		if maxAttempts < numSamples {
			return "", 0, fmt.Errorf("-grammar-max-attempts=%d must be >= -num-samples=%d", maxAttempts, numSamples)
		}
		return policy, maxAttempts, nil
	default:
		return "", 0, fmt.Errorf("-grammar-on-incomplete=%q must be error or skip", raw)
	}
}

func generationAttemptLimitError(plan generationPlan, stats generationRunStats) error {
	return fmt.Errorf(
		"generated %d of %d requested samples after %d attempts; skipped %d incomplete grammar outputs (maximum attempts %d reached)",
		stats.Completed, plan.numSamples, stats.Attempts, stats.SkippedIncomplete, plan.effectiveMaxAttempts(),
	)
}

func validateGenerationIncompleteProcessor(plan generationPlan, factory LogitProcessorFactory) error {
	if plan.skipsIncomplete() && factory == nil {
		return fmt.Errorf("-grammar-on-incomplete=skip requires -grammar-table, -grammar, or -grammar-string")
	}
	return nil
}

func writeGenerationSkipSummary(w io.Writer, stats generationRunStats) {
	if w == nil {
		return
	}
	_, _ = fmt.Fprintf(
		w,
		"generated %d samples, skipped %d incomplete (grammar not accepting at max-tokens or seq_len)\n",
		stats.Completed, stats.SkippedIncomplete,
	)
}
