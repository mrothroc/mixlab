package train

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
