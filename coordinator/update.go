package coordinator

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Accept is called only after workload/attempt authorization. A durable receipt
// and the accepted set are one repository transaction; arrival order is ignored
// by aggregation. An exact retry after advancement returns the original receipt.
func (m Model) Accept(ctx context.Context, id string, u learner.Update) (learner.Receipt, error) {
	s, e := m.load(ctx, id)
	if e != nil {
		return learner.Receipt{}, e
	}
	return m.accept(ctx, s, u)
}

// AcceptSubmission checks the operational fence on the SAME repository revision
// as update publication. HTTP/relay adapters must use this rather than a separate
// CheckSubmission followed by Accept, which could race attempt replacement.
func (m Model) AcceptSubmission(ctx context.Context, q learner.Submission) (learner.Receipt, error) {
	s, err := m.load(ctx, q.Update.Binding.RunID)
	if err != nil {
		return learner.Receipt{}, err
	}
	if err = s.CheckSubmission(q); err != nil {
		return learner.Receipt{}, err
	}
	return m.accept(ctx, s, q.Update)
}
func (m Model) accept(ctx context.Context, s State, u learner.Update) (learner.Receipt, error) {
	id := s.Plan.Cohort.RunID
	var e error
	if e = u.Validate(); e != nil {
		return learner.Receipt{}, e
	}
	if s.Commit == nil || u.Binding != s.Commit.Binding {
		return learner.Receipt{}, fmt.Errorf("update requires matching committed run")
	}
	if e = s.checkRecoveryWrite(u.WorkerID); e != nil {
		return learner.Receipt{}, e
	}
	index := -1
	for i, r := range s.Commit.Registrations {
		if u.WorkerID == r.Slot.Member.WorkerID && u.ParticipantID == r.Slot.Member.ParticipantID && u.IslandID == r.Slot.Member.IslandID {
			index = i
		}
	}
	if index < 0 {
		return learner.Receipt{}, fmt.Errorf("update is outside committed cohort")
	}
	h, e := run.Digest(u)
	if e != nil {
		return learner.Receipt{}, e
	}
	old, e := m.Repository.Receipt(ctx, id, u.ID)
	if e == nil {
		if old.UpdateHash == h && old.Binding == u.Binding {
			return old, nil
		}
		return learner.Receipt{}, ErrConflict
	}
	if !errors.Is(e, ErrNotFound) {
		return learner.Receipt{}, e
	}
	if s.Phase != Open || u.Round != s.Round {
		return learner.Receipt{}, fmt.Errorf("stale round or round not open")
	}
	a := s.Assignments[index]
	if e = u.ValidateFor(a); e != nil {
		return learner.Receipt{}, e
	}
	if m.Clock.Unix() < u.FinishedAt || m.Clock.Unix() >= a.ExpiresAt {
		return learner.Receipt{}, fmt.Errorf("update outside active deadline")
	}
	tokens := u.EffectiveTokens
	for _, v := range s.Updates {
		if v.WorkerID == u.WorkerID {
			return learner.Receipt{}, ErrConflict
		}
		if v.EffectiveTokens > ^uint64(0)-tokens {
			return learner.Receipt{}, fmt.Errorf("effective token count overflow")
		}
		tokens += v.EffectiveTokens
	}
	if e = m.Allocations.Ready(ctx, s.Plan.Slots[index]); e != nil {
		return learner.Receipt{}, e
	}
	base, e := m.Artifacts.Load(ctx, s.Base)
	if e != nil {
		return learner.Receipt{}, e
	}
	delta, e := m.Artifacts.Load(ctx, u.Delta)
	if e != nil {
		return learner.Receipt{}, e
	}
	if e = optimizer.ValidateMatch(base, delta); e != nil {
		return learner.Receipt{}, e
	}
	if uint64(len(delta)) != u.TensorCount {
		return learner.Receipt{}, fmt.Errorf("delta tensor count differs from manifest")
	}
	var sum, maxAbs float64
	for _, t := range delta {
		for _, v := range t.Values {
			x := float64(v)
			sum += x * x
			maxAbs = math.Max(maxAbs, math.Abs(x))
		}
	}
	close := func(a, b float64) bool { return math.Abs(a-b) <= 1e-6*math.Max(1, math.Max(math.Abs(a), math.Abs(b))) }
	if !close(math.Sqrt(sum), u.DeltaL2) || !close(maxAbs, u.DeltaMaxAbs) {
		return learner.Receipt{}, fmt.Errorf("delta summaries differ from verified tensors")
	}
	receipt := learner.Receipt{Format: "mixlab_update_receipt_v1", Binding: u.Binding, UpdateID: u.ID, UpdateHash: h, PrivateHash: u.PrivateHash}
	next := s
	next.Updates = append(append([]learner.Update(nil), s.Updates...), u)
	if e = m.save(ctx, s, Change{Next: next, Receipt: &receipt}); e != nil {
		return learner.Receipt{}, e
	}
	return receipt, nil
}
