package coordinator

import (
	"context"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

func TestStartCommitmentExactRetryAndBinding(t *testing.T) {
	h := setup(t)
	a := h.start(t)[0]
	ctx := context.Background()
	ah, _ := run.Digest(a)
	s := learner.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: ah, PrivateHash: h.plan.Contract.ConfigHash}
	h.repo.failAfter = true
	if err := h.m.RecordStart(ctx, s); err == nil {
		t.Fatal("expected lost response")
	}
	if err := h.m.RecordStart(ctx, s); err != nil {
		t.Fatal(err)
	}
	if len(h.state(t).Starts) != 1 {
		t.Fatal("duplicate start")
	}
	s.PrivateHash = h.plan.SHA256
	if err := h.m.RecordStart(ctx, s); err == nil {
		t.Fatal("changed replay state accepted")
	}
	s.WorkerID = "other"
	if err := h.m.RecordStart(ctx, s); err == nil {
		t.Fatal("unassigned worker accepted")
	}
}

func TestRoundZeroRecoveryRequiresCommittedStartAndFencesOldAttempt(t *testing.T) {
	h := setup(t)
	a := h.start(t)[0]
	ctx := context.Background()
	r := h.commit.Registrations[0]
	ah, _ := run.Digest(a)
	start := learner.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: ah, PrivateHash: strings.Repeat("d", 64)}
	next := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: a.Binding, WorkerID: a.WorkerID, PreviousAttemptID: r.AttemptID, AttemptID: "recovered", WorkloadGeneration: r.WorkloadGeneration + 1, Allocation: r.Slot.Allocation, NodeAuthorizationHash: strings.Repeat("a", 64), PrivateHash: start.PrivateHash, PrecisionHash: r.PrecisionHash}
	allow := recoveryCheck(func(context.Context, run.AttemptBinding) error { return nil })
	if h.m.BindAttempt(ctx, next, allow) == nil {
		t.Fatal("round-zero recovery invented an uncommitted start")
	}
	q := learner.StartSubmission{Format: "mixlab_start_submission_v1", Start: start, PrincipalID: "workload", AttemptID: r.AttemptID, WorkloadGeneration: r.WorkloadGeneration, Allocation: r.Slot.Allocation}
	if err := h.m.RecordStartSubmission(ctx, q); err != nil {
		t.Fatal(err)
	}
	if err := h.m.BindAttempt(ctx, next, allow); err != nil {
		t.Fatal(err)
	}
	if h.m.RecordStartSubmission(ctx, q) == nil {
		t.Fatal("replaced attempt replayed a start commitment")
	}
	q.AttemptID, q.WorkloadGeneration = next.AttemptID, next.WorkloadGeneration
	if err := h.m.RecordStartSubmission(ctx, q); err != nil {
		t.Fatal(err)
	}
	q.Allocation.LeaseID = "wrong"
	if h.m.RecordStartSubmission(ctx, q) == nil {
		t.Fatal("start replay bypassed allocation binding")
	}
}

func TestStartStateRejectsCorruptOrDuplicateCommitment(t *testing.T) {
	h := setup(t)
	a := h.start(t)[0]
	ah, _ := run.Digest(a)
	s := h.state(t)
	start := learner.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: ah, PrivateHash: h.plan.Contract.ConfigHash}
	s.Starts = []learner.Start{start, start}
	if s.Validate() == nil {
		t.Fatal("duplicate commitment")
	}
	s.Starts = s.Starts[:1]
	s.Starts[0].AssignmentHash = h.plan.SHA256
	if s.Validate() == nil {
		t.Fatal("unbound commitment")
	}
}
