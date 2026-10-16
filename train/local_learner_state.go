package train

import (
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
)

const localLearnerSnapshotVersion = "mixlab_local_learner_v1"

func (s *LocalLearnerSession) Snapshot() (LocalLearnerSnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.available(); e != nil {
		return LocalLearnerSnapshot{}, e
	}
	w, e := s.engine.weights()
	if e != nil {
		return LocalLearnerSnapshot{}, e
	}
	o, e := s.engine.state()
	if e != nil {
		return LocalLearnerSnapshot{}, e
	}
	out := LocalLearnerSnapshot{Format: localLearnerSnapshotVersion, Contract: s.contract, Precision: s.policy, Sampler: s.loader.State(), Model: localModel(w, s.shapes), Optimizer: o, Attempt: s.attempt, EffectiveTokens: s.tokens, EffectiveExamples: s.examples}
	if e = s.validateSnapshot(out); e != nil {
		return LocalLearnerSnapshot{}, e
	}
	return out, nil
}

func (s *LocalLearnerSession) validateSnapshot(v LocalLearnerSnapshot) error {
	if v.Format != localLearnerSnapshotVersion || v.Contract != s.contract || v.Precision != s.policy || v.Sampler.Identity != s.loader.State().Identity || v.Attempt > uint64(s.cfg.Training.Steps) {
		return fmt.Errorf("local snapshot contract/identity/precision mismatch")
	}
	o := v.Optimizer.Optimizer
	if o.AttemptedSteps != v.Attempt || o.CommittedSteps != v.Attempt || o.SkippedSteps != 0 || o.ConsecutiveSkipped != 0 || o.LastStepSkipped || o.LastLossNonfinite != 0 || o.LastGradientNonfinite != 0 || o.LastStateNonfinite != 0 {
		return fmt.Errorf("local snapshot is not a clean optimizer boundary")
	}
	k := uint64(s.cfg.Training.Distributed.GradientAccumulationSteps)
	wantExamples := v.Attempt * k * uint64(s.cfg.Training.BatchTokens/s.cfg.SeqLen)
	maxTokens := v.Attempt * k * uint64(s.cfg.Training.BatchTokens)
	if v.EffectiveExamples != wantExamples || v.EffectiveTokens > maxTokens || (v.Attempt == 0 && v.EffectiveTokens != 0) || (v.Attempt > 0 && v.EffectiveTokens == 0) {
		return fmt.Errorf("invalid local effective counters")
	}
	if e := s.loader.ValidateConsumedRows(v.Sampler, wantExamples); e != nil {
		return e
	}
	for _, t := range v.Optimizer.Tensors {
		for _, x := range t.Data {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || (t.Kind == gpu.OptimizerStateAdamV && x < 0) {
				return fmt.Errorf("invalid optimizer moment")
			}
		}
	}
	_, e := localModelData(v.Model, s.shapes)
	return e
}

// Restore prepares a separate engine before replacing the current one. Failed
// model/moment/cursor validation cannot partially overwrite live private state.
func (s *LocalLearnerSession) Restore(v LocalLearnerSnapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.engine == nil {
		return fmt.Errorf("local learner closed")
	}
	if e := s.validateSnapshot(v); e != nil {
		return e
	}
	w, e := localModelData(v.Model, s.shapes)
	if e != nil {
		return e
	}
	candidate, e := s.factory(w, false)
	if e != nil {
		return e
	}
	step := int(v.Attempt) * s.cfg.Training.Distributed.GradientAccumulationSteps
	if e = candidate.restore(v.Optimizer, step); e != nil {
		candidate.close()
		return e
	}
	if e = s.loader.Restore(v.Sampler); e != nil {
		candidate.close()
		return e
	}
	old := s.engine
	s.engine = candidate
	s.attempt, s.tokens, s.examples = v.Attempt, v.EffectiveTokens, v.EffectiveExamples
	s.failed = false
	old.close()
	return nil
}

// EvaluateGlobal uses an isolated dropout-free engine. Training weights, AdamW
// moments, counters, sampler cursor and keyed RNG never enter its mutable state.
func (s *LocalLearnerSession) EvaluateGlobal(model []LocalLearnerTensor, batches []data.Batch) (float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.available(); e != nil {
		return 0, e
	}
	w, e := localModelData(model, s.shapes)
	if e != nil {
		return 0, e
	}
	eval, e := s.factory(w, true)
	if e != nil {
		return 0, e
	}
	defer eval.close()
	var numerator, denominator float64
	for _, b := range batches {
		loss, n, e := evaluateLocalBatch(s.cfg, eval, b, int(s.attempt))
		if e != nil {
			return 0, e
		}
		numerator += loss * float64(n)
		denominator += float64(n)
	}
	if denominator <= 0 {
		return 0, fmt.Errorf("global evaluation has no valid targets")
	}
	return numerator / denominator, nil
}
