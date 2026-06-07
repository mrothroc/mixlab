package train

import (
	"fmt"
	"math"
)

// applyTrainingSWAOverrides applies any CLI SWA/EMA overrides onto the config's
// training spec, validating each provided value, and returns human-readable log
// lines describing the overrides that were applied.
func applyTrainingSWAOverrides(cfg *ArchConfig, opts TrainOptions) ([]string, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	var logs []string
	if opts.SWAStartOverride != nil {
		if *opts.SWAStartOverride < 0 {
			return nil, fmt.Errorf("invalid -swa-start=%d (must be >= 0)", *opts.SWAStartOverride)
		}
		cfg.Training.SWAStart = *opts.SWAStartOverride
		logs = append(logs, fmt.Sprintf("training.swa_start overridden by CLI: %d", cfg.Training.SWAStart))
	}
	if opts.SWADecayOverride != nil {
		v := *opts.SWADecayOverride
		if math.IsNaN(float64(v)) || v < 0 || v >= 1 {
			return nil, fmt.Errorf("invalid -swa-decay=%g (must be in [0,1))", v)
		}
		cfg.Training.SWADecay = v
		logs = append(logs, fmt.Sprintf("training.swa_decay overridden by CLI: %g", cfg.Training.SWADecay))
	}
	if opts.SWAIntervalOverride != nil {
		if *opts.SWAIntervalOverride <= 0 {
			return nil, fmt.Errorf("invalid -swa-interval=%d (must be > 0)", *opts.SWAIntervalOverride)
		}
		cfg.Training.SWAInterval = *opts.SWAIntervalOverride
		logs = append(logs, fmt.Sprintf("training.swa_interval overridden by CLI: %d", cfg.Training.SWAInterval))
	}
	return logs, nil
}

// shouldUpdateSWA reports whether the current training step should contribute
// an SWA snapshot given the configured start step and update interval.
func shouldUpdateSWA(step, start, interval int) bool {
	return start > 0 && interval > 0 && step >= start && (step-start)%interval == 0
}

// hasSWAWeights reports whether any per-weight EMA slot has been populated.
func hasSWAWeights(ema [][]float32) bool {
	for _, weight := range ema {
		if len(weight) != 0 {
			return true
		}
	}
	return false
}

// updateEMAWeights mixes the current trainer weights into the running EMA in
// place. The first time a slot is observed it is copied directly; subsequent
// updates take a (decay, 1-decay) weighted average.
func updateEMAWeights(ema, current [][]float32, decay float32) {
	oneMinusDecay := 1 - decay
	for i, weight := range current {
		if len(ema[i]) == 0 {
			ema[i] = append([]float32(nil), weight...)
			continue
		}
		for j, value := range weight {
			ema[i][j] = decay*ema[i][j] + oneMinusDecay*value
		}
	}
}

// cloneWeights returns an independent copy of a weight slice-of-slices.
func cloneWeights(weights [][]float32) [][]float32 {
	cloned := make([][]float32, len(weights))
	for i, weight := range weights {
		cloned[i] = append([]float32(nil), weight...)
	}
	return cloned
}
