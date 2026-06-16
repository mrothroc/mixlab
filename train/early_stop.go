package train

import (
	"fmt"
	"math"
)

type earlyStopState struct {
	spec     *EarlyStopSpec
	best     float64
	haveBest bool
	stale    int
}

func newEarlyStopState(spec *EarlyStopSpec) *earlyStopState {
	if spec == nil || (spec.Patience <= 0 && spec.ValGT <= 0) {
		return nil
	}
	return &earlyStopState{spec: spec, best: math.Inf(1)}
}

func (s *earlyStopState) observe(step int, valLoss float64) (bool, string) {
	if s == nil || s.spec == nil {
		return false, ""
	}
	if s.spec.ValGT > 0 && step >= s.spec.AtStep && valLoss > s.spec.ValGT {
		return true, fmt.Sprintf("validation loss %.4f > %.4f at step %d", valLoss, s.spec.ValGT, step)
	}

	improved := !s.haveBest || valLoss < s.best-s.spec.MinDelta
	if improved {
		s.best = valLoss
		s.haveBest = true
		s.stale = 0
		return false, ""
	}
	if s.spec.Patience <= 0 || step < s.spec.MinSteps {
		return false, ""
	}
	s.stale++
	if s.stale >= s.spec.Patience {
		return true, fmt.Sprintf("validation loss did not improve beyond min_delta %.4g for %d validation checks (best=%.4f current=%.4f)", s.spec.MinDelta, s.spec.Patience, s.best, valLoss)
	}
	return false, ""
}
