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

func (s *earlyStopState) resumeSnapshot() resumeEarlyStop {
	if s == nil || s.spec == nil {
		return resumeEarlyStop{}
	}
	snapshot := resumeEarlyStop{
		Enabled:  true,
		HaveBest: s.haveBest,
		Stale:    s.stale,
	}
	if s.haveBest {
		snapshot.Best = s.best
	}
	return snapshot
}

func (s *earlyStopState) restoreResumeSnapshot(snapshot resumeEarlyStop) error {
	if !snapshot.Enabled {
		return nil
	}
	if s == nil || s.spec == nil {
		return fmt.Errorf("checkpoint contains early-stop state but training.early_stop is disabled")
	}
	if snapshot.Stale < 0 || (snapshot.HaveBest && (math.IsNaN(snapshot.Best) || math.IsInf(snapshot.Best, 0))) {
		return fmt.Errorf("checkpoint contains invalid early-stop state")
	}
	s.best = snapshot.Best
	s.haveBest = snapshot.HaveBest
	s.stale = snapshot.Stale
	return nil
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
