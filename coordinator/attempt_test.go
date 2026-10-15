package coordinator

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

type recoveryCheck func(context.Context, run.AttemptBinding) error

func (f recoveryCheck) Check(c context.Context, a run.AttemptBinding) error { return f(c, a) }
func TestAdmissionAbortCommitRace(t *testing.T) {
	ctx := context.Background()
	for _, commitFirst := range []bool{false, true} {
		h := setup(t)
		if err := h.m.Stage(ctx, h.plan, h.policy); err != nil {
			t.Fatal(err)
		}
		if commitFirst {
			if err := h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); err != nil {
				t.Fatal(err)
			}
		}
		c, err := h.m.AbortAdmission(ctx, h.plan.Cohort.RunID)
		if err != nil {
			t.Fatal(err)
		}
		if commitFirst {
			if c == nil || c.Binding != h.commit.Binding {
				t.Fatal("lost committed run")
			}
		} else {
			if c != nil {
				t.Fatal("unexpected commit")
			}
			if err = h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); err == nil {
				t.Fatal("late commit escaped abort fence")
			}
			if err = h.m.Stage(ctx, h.plan, h.policy); err != nil {
				t.Fatal(err)
			}
			if h.state(t).Phase != Aborted {
				t.Fatal("stage reopened aborted run")
			}
		}
	}
}

func TestAdmissionFenceBeforeDelayedStage(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	h.clock.now.Store(h.plan.ExpiresAt)
	if c, err := h.m.FenceAdmission(ctx, h.plan, h.policy); err != nil || c != nil {
		t.Fatal(c, err)
	}
	if err := h.m.Stage(ctx, h.plan, h.policy); err != nil {
		t.Fatal(err)
	}
	if err := h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); err == nil {
		t.Fatal("late commit escaped pre-stage tombstone")
	}
	if h.state(t).Phase != Aborted {
		t.Fatal("tombstone lost")
	}
}
func TestDurableAttemptFencing(t *testing.T) {
	h := setup(t)
	h.start(t)
	ctx := context.Background()
	s := h.state(t)
	private := strings.Repeat("d", 64)
	s.PrivateHashes[0] = private
	// A completed local checkpoint is prerequisite for recovery, even in round 0.
	s.Assignments[0].PreviousPrivateHash = private
	if err := h.m.save(ctx, h.state(t), Change{Next: s}); err != nil {
		t.Fatal(err)
	}
	r := h.commit.Registrations[0]
	a := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: h.commit.Binding, WorkerID: r.Slot.Member.WorkerID, PreviousAttemptID: r.AttemptID, AttemptID: "new-attempt", WorkloadGeneration: r.WorkloadGeneration + 1, Allocation: r.Slot.Allocation, NodeAuthorizationHash: strings.Repeat("a", 64), PrivateHash: private, PrecisionHash: r.PrecisionHash}
	allow := recoveryCheck(func(context.Context, run.AttemptBinding) error { return nil })
	deny := recoveryCheck(func(context.Context, run.AttemptBinding) error { return errors.New("node revoked") })
	if h.m.BindAttempt(ctx, a, deny) == nil {
		t.Fatal("missing node authorization accepted")
	}
	for _, mutate := range []func(*run.AttemptBinding){func(a *run.AttemptBinding) { a.PreviousAttemptID = "wrong" }, func(a *run.AttemptBinding) { a.PrecisionHash = strings.Repeat("b", 64) }, func(a *run.AttemptBinding) { a.PrivateHash = strings.Repeat("b", 64) }, func(a *run.AttemptBinding) { a.WorkloadGeneration = 1 }} {
		bad := a
		mutate(&bad)
		if h.m.BindAttempt(ctx, bad, allow) == nil {
			t.Fatal("bad retry accepted")
		}
	}
	h.repo.failAfter = true
	if h.m.BindAttempt(ctx, a, allow) == nil {
		t.Fatal("expected lost response")
	}
	if err := h.m.BindAttempt(ctx, a, allow); err != nil {
		t.Fatal(err)
	}
	s = h.state(t)
	if len(s.Attempts) != 1 {
		t.Fatal("duplicate retry binding")
	}
	id, g, err := s.CurrentAttempt(a.WorkerID)
	if err != nil || id != a.AttemptID || g != a.WorkloadGeneration {
		t.Fatal(id, g, err)
	}
	if h.m.BindAttempt(ctx, a, deny) == nil {
		t.Fatal("exact retry bypassed current authorization")
	}
	u := h.update(t, s.Assignments[0], 0)
	q := learner.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: "workload", AttemptID: r.AttemptID, WorkloadGeneration: r.WorkloadGeneration, Allocation: r.Slot.Allocation}
	if _, err := h.m.AcceptSubmission(ctx, q); err == nil {
		t.Fatal("stale submission accepted")
	}
	q.AttemptID = a.AttemptID
	q.WorkloadGeneration = a.WorkloadGeneration
	receipt, err := h.m.AcceptSubmission(ctx, q)
	if err != nil {
		t.Fatal(err)
	}
	// Lost acknowledgement after acceptance: recover against the completed
	// private checkpoint and return the SAME receipt without another update.
	next := a
	next.PreviousAttemptID, next.AttemptID = a.AttemptID, "after-accepted"
	next.WorkloadGeneration++
	if h.m.BindAttempt(ctx, next, allow) == nil {
		t.Fatal("retry accepted old private state after update")
	}
	next.PrivateHash = u.PrivateHash
	if err = h.m.BindAttempt(ctx, next, allow); err != nil {
		t.Fatal(err)
	}
	if _, err = h.m.AcceptSubmission(ctx, q); err == nil {
		t.Fatal("old attempt read receipt after replacement")
	}
	q.AttemptID, q.WorkloadGeneration = next.AttemptID, next.WorkloadGeneration
	again, err := h.m.AcceptSubmission(ctx, q)
	if err != nil || again != receipt || len(h.state(t).Updates) != 1 {
		t.Fatal("recovery duplicated accepted work", again, err)
	}
}

type interleavedArtifacts struct {
	Artifacts
	once   sync.Once
	change func()
}

func (a *interleavedArtifacts) Load(ctx context.Context, b run.Blob) ([]optimizer.Tensor, error) {
	a.once.Do(a.change)
	return a.Artifacts.Load(ctx, b)
}
func TestSubmissionCannotRaceAttemptReplacement(t *testing.T) {
	h := setup(t)
	h.start(t)
	ctx := context.Background()
	s := h.state(t)
	private := strings.Repeat("e", 64)
	s.PrivateHashes[0] = private
	s.Assignments[0].PreviousPrivateHash = private
	if err := h.m.save(ctx, h.state(t), Change{Next: s}); err != nil {
		t.Fatal(err)
	}
	u := h.update(t, s.Assignments[0], 0)
	r := h.commit.Registrations[0]
	q := learner.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: "workload", AttemptID: r.AttemptID, WorkloadGeneration: r.WorkloadGeneration, Allocation: r.Slot.Allocation}
	a := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: h.commit.Binding, WorkerID: r.Slot.Member.WorkerID, PreviousAttemptID: r.AttemptID, AttemptID: "new", WorkloadGeneration: 2, Allocation: r.Slot.Allocation, NodeAuthorizationHash: strings.Repeat("a", 64), PrivateHash: private, PrecisionHash: r.PrecisionHash}
	h.m.Artifacts = &interleavedArtifacts{Artifacts: h.artifacts, change: func() {
		if err := h.m.BindAttempt(ctx, a, recoveryCheck(func(context.Context, run.AttemptBinding) error { return nil })); err != nil {
			t.Fatal(err)
		}
	}}
	if _, err := h.m.AcceptSubmission(ctx, q); !errors.Is(err, ErrConflict) {
		t.Fatal("racing submission escaped fence", err)
	}
	if len(h.state(t).Updates) != 0 {
		t.Fatal("stale update published")
	}
}
