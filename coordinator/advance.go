package coordinator

import (
	"context"
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Advance performs one recoverable round transition. Repository CAS is the
// linearization point; duplicate/concurrent calls cannot advance twice. Artifact
// writes and evaluation requests are idempotent and precede their publication.
func (m Model) Advance(ctx context.Context, id string) error {
	s, e := m.load(ctx, id)
	if e != nil {
		return e
	}
	switch s.Phase {
	case Open:
		if len(s.Updates) != len(s.Plan.Slots) {
			return fmt.Errorf("all committed workers must submit before aggregation")
		}
		for _, slot := range s.Plan.Slots {
			if e = m.Allocations.Ready(ctx, slot); e != nil {
				return e
			}
		}
		next := s
		next.Phase = Aggregating
		return m.save(ctx, s, Change{Next: next})
	case Aggregating:
		return m.aggregate(ctx, s)
	case Publishing:
		return m.publish(ctx, s)
	case Evaluating:
		return m.evaluate(ctx, s)
	case Complete:
		return nil
	default:
		return fmt.Errorf("cannot advance run in phase %s", s.Phase)
	}
}

func (m Model) aggregate(ctx context.Context, s State) error {
	base, e := m.Artifacts.Load(ctx, s.Base)
	if e != nil {
		return e
	}
	momentum, e := m.Artifacts.Load(ctx, s.Momentum)
	if e != nil {
		return e
	}
	order := make([]string, len(s.Plan.Slots))
	updates := make([]optimizer.Contribution, len(order))
	accepted := make([]run.AcceptedUpdate, len(order))
	byID := make(map[string]learner.Update, len(s.Updates))
	for _, u := range s.Updates {
		if _, ok := byID[u.WorkerID]; ok {
			return fmt.Errorf("duplicate accepted worker")
		}
		byID[u.WorkerID] = u
	}
	for i, slot := range s.Plan.Slots {
		u, ok := byID[slot.Member.WorkerID]
		if !ok {
			return fmt.Errorf("accepted cohort incomplete")
		}
		if e = u.ValidateFor(s.Assignments[i]); e != nil {
			return e
		}
		delta, e := m.Artifacts.Load(ctx, u.Delta)
		if e != nil {
			return e
		}
		order[i] = u.WorkerID
		updates[i] = optimizer.Contribution{WorkerID: u.WorkerID, Delta: delta, Attempted: u.Attempted, Committed: u.Committed, Skipped: u.Skipped, EffectiveTokens: u.EffectiveTokens}
		h, e := run.Digest(u)
		if e != nil {
			return e
		}
		accepted[i] = run.AcceptedUpdate{WorkerID: u.WorkerID, UpdateID: u.ID, UpdateHash: h, PrivateHash: u.PrivateHash, PrecisionHash: u.PrecisionHash}
	}
	o := s.Plan.Outer
	result, e := optimizer.Step(optimizer.Config{LR: o.LR, Momentum: o.Momentum, Aggregation: o.Aggregation, InnerSteps: o.InnerSteps}, order, base, momentum, updates)
	if e != nil {
		return e
	}
	model, e := m.Artifacts.Put(ctx, "model_f32", result.Model)
	if e != nil {
		return e
	}
	mom, e := m.Artifacts.Put(ctx, "outer_state_f32", result.Momentum)
	if e != nil {
		return e
	}
	r := run.Round{Format: "mixlab_distributed_round_v1", Binding: s.Commit.Binding, Round: s.Round, Contract: s.Plan.Contract, Model: model, OuterState: mom, Outer: o, Accepted: accepted, EffectiveTokens: result.EffectiveTokens, CreatedAt: m.Clock.Unix()}
	r.SHA256, e = r.Digest()
	if e != nil {
		return e
	}
	if e = r.ValidateFor(s.Plan, *s.Commit); e != nil {
		return e
	}
	next := s
	next.Phase = Publishing
	next.Candidate = &r
	return m.save(ctx, s, Change{Next: next})
}

func (m Model) publish(ctx context.Context, s State) error {
	if s.Candidate == nil {
		return fmt.Errorf("missing candidate round")
	}
	if e := s.Candidate.ValidateFor(s.Plan, *s.Commit); e != nil {
		return e
	}
	p := s.Policy
	now := m.Clock.Unix()
	if now <= 0 || now > math.MaxInt64-p.TimeoutSeconds {
		return fmt.Errorf("invalid evaluation deadline")
	}
	var registration *run.Registration
	for i := range s.Commit.Registrations {
		r := &s.Commit.Registrations[i]
		if r.Slot.Member.WorkerID == p.WorkerID {
			registration = r
		}
	}
	if registration == nil {
		return fmt.Errorf("evaluator is not committed")
	}
	a := run.EvaluationAssignment{Format: "mixlab_evaluation_assignment_v1", ID: fmt.Sprintf("evaluation-%d", s.Round), Binding: s.Commit.Binding, Coordinator: s.Plan.Coordinator, Round: s.Round, Model: s.Candidate.Model, ValidationHash: p.ValidationHash, Split: p.Split, BatchTokens: p.BatchTokens, SequenceLength: p.SequenceLength, Batches: p.Batches, WorkerID: p.WorkerID, Precision: registration.Precision, PrecisionHash: registration.PrecisionHash, IssuedAt: now, ExpiresAt: now + p.TimeoutSeconds}
	if e := a.Validate(); e != nil {
		return e
	}
	next := s
	next.Phase = Evaluating
	next.Base = s.Candidate.Model
	next.Momentum = s.Candidate.OuterState
	next.Evaluation = &a
	next.PrivateHashes = make([]string, len(s.Candidate.Accepted))
	for i, u := range s.Candidate.Accepted {
		next.PrivateHashes[i] = u.PrivateHash
	}
	return m.save(ctx, s, Change{Next: next, Published: s.Candidate})
}

func (m Model) evaluate(ctx context.Context, s State) error {
	if s.Evaluation == nil {
		return fmt.Errorf("missing evaluation assignment")
	}
	a := *s.Evaluation
	if m.Clock.Unix() >= a.ExpiresAt {
		return fmt.Errorf("evaluation deadline expired")
	}
	for _, slot := range s.Plan.Slots {
		if e := m.Allocations.Ready(ctx, slot); e != nil {
			return e
		}
	}
	result, e := m.Evaluator.Evaluate(ctx, a)
	if e != nil {
		return e
	}
	if e = result.Validate(); e != nil {
		return e
	}
	if result.Assignment != a || m.Clock.Unix() < result.FinishedAt || m.Clock.Unix() >= a.ExpiresAt {
		return fmt.Errorf("evaluator returned a different/expired assignment")
	}
	for i, slot := range s.Plan.Slots {
		if slot.Member.WorkerID == a.WorkerID && result.PrivateBeforeHash != s.PrivateHashes[i] {
			return fmt.Errorf("evaluation did not preserve expected completed private state")
		}
	}
	h, e := run.Digest(result)
	if e != nil {
		return e
	}
	next := s
	next.Round++
	next.Phase = Ready
	decision := run.EvaluationDecision{Format: "mixlab_evaluation_decision_v1", Binding: s.Commit.Binding, Round: s.Round, ResultHash: h, Decision: "continue"}
	if next.Round == s.Plan.Outer.TotalSteps/s.Plan.Outer.InnerSteps || (s.Policy.TargetLoss != nil && result.Loss <= *s.Policy.TargetLoss) {
		next.Phase = Complete
		decision.Decision = "complete"
	}
	return m.save(ctx, s, Change{Next: next, Evaluation: &result, Decision: &decision})
}
