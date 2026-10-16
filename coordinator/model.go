package coordinator

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

const maxSeconds int64 = 7 * 24 * 3600

func (m Model) load(ctx context.Context, id string) (State, error) {
	if m.Repository == nil || m.Artifacts == nil || m.Allocations == nil || m.Clock == nil || m.Evaluator == nil {
		return State{}, fmt.Errorf("coordinator ports required")
	}
	s, e := m.Repository.Load(ctx, id)
	if e != nil {
		return s, e
	}
	if s.Plan.Cohort.RunID != id {
		return State{}, fmt.Errorf("repository returned another run")
	}
	if e = s.Validate(); e != nil {
		return State{}, e
	}
	return s, nil
}
func (m Model) save(ctx context.Context, s State, c Change) error {
	if s.Revision == ^uint64(0) {
		return fmt.Errorf("revision overflow")
	}
	c.Next.Revision = s.Revision + 1
	if e := c.Next.Validate(); e != nil {
		return e
	}
	return m.Repository.Save(ctx, s.Plan.Cohort.RunID, s.Revision, c)
}

// Stage persists only a proposal. No round can open until Commit succeeds.
func (m Model) Stage(ctx context.Context, p run.Plan, policy EvaluationPolicy) error {
	if e := p.Validate(); e != nil {
		return e
	}
	s, e := m.load(ctx, p.Cohort.RunID)
	if e == nil {
		h, _ := run.Digest(s.Policy)
		want, _ := run.Digest(policy)
		if s.Plan.SHA256 == p.SHA256 && h == want {
			return nil
		}
		return ErrConflict
	}
	if !errors.Is(e, ErrNotFound) {
		return e
	}
	now := m.Clock.Unix()
	if now < p.CreatedAt || now >= p.ExpiresAt {
		return fmt.Errorf("staged plan outside admission deadline")
	}
	if e = validatePolicy(policy, p); e != nil {
		return e
	}
	s = State{Plan: p}
	next := State{Phase: Staged, Plan: p, Policy: policy}
	return m.save(ctx, s, Change{Next: next})
}

func validatePolicy(p EvaluationPolicy, plan run.Plan) error {
	found := false
	for _, s := range plan.Slots {
		if s.Member.WorkerID == p.WorkerID {
			found = true
		}
	}
	if !found || p.TimeoutSeconds <= 0 || p.TimeoutSeconds > maxSeconds || p.BatchTokens == 0 || p.SequenceLength == 0 || p.BatchTokens%p.SequenceLength != 0 || p.Batches == 0 || p.Split == "" || len(p.Split) > 256 || strings.IndexFunc(p.Split, func(r rune) bool { return r < 33 || r > 126 }) >= 0 {
		return fmt.Errorf("invalid designated evaluator policy")
	}
	if p.TargetLoss != nil && (math.IsNaN(*p.TargetLoss) || math.IsInf(*p.TargetLoss, 0) || *p.TargetLoss < 0) {
		return fmt.Errorf("invalid target loss")
	}
	ref := plan.BaseModel
	ref.Ref.SHA256 = p.ValidationHash
	return ref.Validate()
}

func (m Model) Commit(ctx context.Context, id string, c run.Commit) error {
	s, e := m.load(ctx, id)
	if e != nil {
		return e
	}
	if e = c.ValidateFor(s.Plan); e != nil {
		return e
	}
	if s.Commit != nil {
		if s.Commit.Binding == c.Binding {
			return nil
		}
		return ErrConflict
	}
	if s.Phase != Staged || m.Clock.Unix() >= s.Plan.ExpiresAt || m.Clock.Unix() < c.CommittedAt {
		return fmt.Errorf("run cannot commit outside admission")
	}
	base, e := m.Artifacts.Load(ctx, s.Plan.BaseModel)
	if e != nil {
		return e
	}
	zeros, e := optimizer.ZerosLike(base)
	if e != nil {
		return e
	}
	momentum, e := m.Artifacts.Put(ctx, "outer_state_f32", zeros)
	if e != nil {
		return e
	}
	if e = checkBlob(momentum, "outer_state_f32"); e != nil {
		return e
	}
	next := s
	next.Commit = &c
	next.Phase = Ready
	next.Base = s.Plan.BaseModel
	next.Momentum = momentum
	next.PrivateHashes = make([]string, len(s.Plan.Slots))
	return m.save(ctx, s, Change{Next: next})
}

// AbortAdmission fences a late commit with the same repository CAS used by
// Commit. A committed result must be handed off, never compensated as unstarted.
func (m Model) AbortAdmission(ctx context.Context, id string) (*run.Commit, error) {
	s, err := m.load(ctx, id)
	if err != nil {
		return nil, err
	}
	if s.Commit != nil {
		return s.Commit, nil
	}
	if s.Phase == Aborted {
		return nil, nil
	}
	if s.Phase != Staged {
		return nil, fmt.Errorf("run is not staged")
	}
	next := s
	next.Phase = Aborted
	return nil, m.save(ctx, s, Change{Next: next})
}

// FenceAdmission also handles a Stage request with an unknown outcome. The
// absent-run tombstone competes on the same CAS as Stage, so delayed staging
// cannot resurrect a compensated run. Unlike Stage, fencing is allowed after
// the admission deadline.
func (m Model) FenceAdmission(ctx context.Context, p run.Plan, policy EvaluationPolicy) (*run.Commit, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	if err := validatePolicy(policy, p); err != nil {
		return nil, err
	}
	s, err := m.load(ctx, p.Cohort.RunID)
	if errors.Is(err, ErrNotFound) {
		return nil, m.save(ctx, State{Plan: p}, Change{Next: State{Phase: Aborted, Plan: p, Policy: policy}})
	}
	if err != nil {
		return nil, err
	}
	h, _ := run.Digest(s.Policy)
	want, _ := run.Digest(policy)
	if s.Plan.SHA256 != p.SHA256 || h != want {
		return nil, ErrConflict
	}
	return m.AbortAdmission(ctx, p.Cohort.RunID)
}

// OpenRound is idempotent for an already open round. Assignments share a fixed
// deadline and are stable across acquire retries; expired work is never extended.
func (m Model) OpenRound(ctx context.Context, id string, seconds int64) ([]run.Assignment, error) {
	s, e := m.load(ctx, id)
	if e != nil {
		return nil, e
	}
	if s.Phase == Open {
		if m.Clock.Unix() >= s.Assignments[0].ExpiresAt {
			return nil, fmt.Errorf("round deadline expired")
		}
		for _, slot := range s.Plan.Slots {
			if e = m.Allocations.Ready(ctx, slot); e != nil {
				return nil, e
			}
		}
		return s.Assignments, nil
	}
	if s.Phase != Ready || s.Commit == nil {
		return nil, fmt.Errorf("committed ready run required")
	}
	now := m.Clock.Unix()
	if seconds <= 0 || seconds > maxSeconds || now <= 0 || now > math.MaxInt64-seconds || now < s.Commit.CommittedAt {
		return nil, fmt.Errorf("invalid round deadline")
	}
	if s.Round >= s.Plan.Outer.TotalSteps/s.Plan.Outer.InnerSteps {
		return nil, fmt.Errorf("no remaining rounds")
	}
	for _, slot := range s.Plan.Slots {
		if e = m.Allocations.Ready(ctx, slot); e != nil {
			return nil, e
		}
	}
	next := s
	next.Phase = Open
	next.Assignments = make([]run.Assignment, len(s.Plan.Slots))
	next.Updates = nil
	next.Starts = nil
	next.Candidate = nil
	next.Evaluation = nil
	for i, r := range s.Commit.Registrations {
		a := run.Assignment{Format: run.AssignmentVersion, ID: fmt.Sprintf("round-%d-worker-%d", s.Round, i), Binding: s.Commit.Binding, Coordinator: s.Plan.Coordinator, WorkerID: r.Slot.Member.WorkerID, Round: s.Round, BaseRound: s.Round, BaseModel: s.Base, OuterState: s.Momentum, Contract: s.Plan.Contract, PartitionID: r.Slot.Member.DatasetPartitionID, InnerSteps: s.Plan.Outer.InnerSteps, PreviousPrivateHash: s.PrivateHashes[i], PrecisionHash: r.PrecisionHash, IssuedAt: now, ExpiresAt: now + seconds}
		if e = a.ValidateFor(s.Plan, *s.Commit); e != nil {
			return nil, e
		}
		next.Assignments[i] = a
	}
	if e = m.save(ctx, s, Change{Next: next}); e != nil {
		return nil, e
	}
	return next.Assignments, nil
}

func checkBlob(b run.Blob, kind string) error {
	if e := b.Validate(); e != nil {
		return e
	}
	if b.Kind != kind {
		return fmt.Errorf("artifact adapter returned wrong kind")
	}
	return nil
}

func (m Model) Fail(ctx context.Context, id, reason string) error {
	if reason == "" || len(reason) > 4096 {
		return fmt.Errorf("bounded failure reason required")
	}
	s, e := m.load(ctx, id)
	if e != nil {
		return e
	}
	if s.Phase == Failed {
		return nil
	}
	if s.Commit == nil || s.Phase == Complete {
		return fmt.Errorf("only active committed runs can fail")
	}
	next := s
	next.Phase = Failed
	next.Failure = reason
	return m.save(ctx, s, Change{Next: next})
}
