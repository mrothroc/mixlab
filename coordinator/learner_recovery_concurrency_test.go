package coordinator

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

type recoveryInterleavedRepository struct {
	Repository
	once   sync.Once
	before func()
}

func (r *recoveryInterleavedRepository) Save(ctx context.Context, id string, revision uint64, c Change) error {
	r.once.Do(r.before)
	return r.Repository.Save(ctx, id, revision, c)
}

func recoveryWriteRequests(t *testing.T, h *harness, a run.Assignment, index int) (learner.StartSubmission, learner.Submission) {
	t.Helper()
	reg := h.commit.Registrations[index]
	ah, err := run.Digest(a)
	if err != nil {
		t.Fatal(err)
	}
	start := learner.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: ah, PrivateHash: strings.Repeat("b", 64)}
	u := h.update(t, a, index)
	u.PrivateHash = strings.Repeat("c", 64)
	return learner.StartSubmission{Format: "mixlab_start_submission_v1", Start: start, PrincipalID: "workload", AttemptID: reg.AttemptID, WorkloadGeneration: reg.WorkloadGeneration, Allocation: reg.Slot.Allocation}, learner.Submission{Format: "mixlab_update_submission_v1", Update: u, PrincipalID: "workload", AttemptID: reg.AttemptID, WorkloadGeneration: reg.WorkloadGeneration, Allocation: reg.Slot.Allocation}
}

func TestLearnerRecoveryFencesLateAndInflightWrites(t *testing.T) {
	for _, operation := range []string{"start", "submit"} {
		for _, timing := range []string{"after_begin", "inflight_before_begin"} {
			t.Run(operation+"/"+timing, func(t *testing.T) {
				h := setup(t)
				ctx := context.Background()
				id := h.plan.Cohort.RunID
				a := h.start(t)
				h.acceptAll(t, a)
				for h.state(t).Phase != Ready {
					h.advance(t)
				}
				a, err := h.m.OpenRound(ctx, id, 100)
				if err != nil {
					t.Fatal(err)
				}
				start, submit := recoveryWriteRequests(t, h, a[0], 0)
				if operation == "submit" {
					if err = h.m.RecordStartSubmission(ctx, start); err != nil {
						t.Fatal(err)
					}
				}
				write := func(m Model) error {
					if operation == "start" {
						return m.RecordStartSubmission(ctx, start)
					}
					_, err := m.AcceptSubmission(ctx, submit)
					return err
				}
				var intent LearnerRecovery
				begin := func() {
					if err := h.m.BeginLearnerRecovery(ctx, id, a[0].WorkerID); err != nil {
						t.Fatal(err)
					}
					intent = *h.state(t).Recovery
				}
				if timing == "inflight_before_begin" {
					// The old request has validated and built its write before Begin
					// commits. Its stale publication must lose CAS, including receipt.
					inflight := h.m
					inflight.Repository = &recoveryInterleavedRepository{Repository: h.repo, before: begin}
					if err = write(inflight); !errors.Is(err, ErrConflict) {
						t.Fatal("inflight write did not lose CAS", err)
					}
				} else {
					begin()
				}
				fenced := h.state(t)
				if len(fenced.Updates) != 0 || operation == "start" && len(fenced.Starts) != 0 {
					t.Fatal("old request published after Begin")
				}
				if _, err = h.repo.Receipt(ctx, id, submit.Update.ID); !errors.Is(err, ErrNotFound) {
					t.Fatal("old request published a receipt", err)
				}
				// Reconstruct the model: the write fence must be durable state,
				// not an in-memory lock or a transient caller flag.
				restarted := Model{Repository: h.repo, Artifacts: h.artifacts, Allocations: h.alloc, Clock: h.clock, Evaluator: h.eval}
				for range 2 {
					if err = write(restarted); err == nil || errors.Is(err, ErrConflict) {
						t.Fatal("late old attempt must receive permanent rejection", err)
					}
				}
				if !reflect.DeepEqual(fenced, h.state(t)) {
					t.Fatal("rejected replay mutated recovery state")
				}
				otherStart, otherSubmit := recoveryWriteRequests(t, h, a[1], 1)
				if err = restarted.RecordStartSubmission(ctx, otherStart); err != nil {
					t.Fatal("unaffected worker cannot start", err)
				}
				if _, err = restarted.AcceptSubmission(ctx, otherSubmit); err != nil {
					t.Fatal("unaffected worker cannot submit", err)
				}
				s := h.state(t)
				private, err := s.ExpectedPrivate(a[0].WorkerID)
				if err != nil || private != intent.PrivateHash || *s.Recovery != intent {
					t.Fatal("recovery private head or intent changed", err)
				}
				reg := h.commit.Registrations[0]
				binding := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: h.commit.Binding, WorkerID: a[0].WorkerID, PreviousAttemptID: intent.Previous, AttemptID: intent.Attempt, WorkloadGeneration: reg.WorkloadGeneration + 1, Allocation: reg.Slot.Allocation, NodeAuthorizationHash: h.plan.SHA256, PrivateHash: intent.PrivateHash, PrecisionHash: reg.PrecisionHash}
				if err = restarted.BindAttempt(ctx, binding, recoveryCheck(func(context.Context, run.AttemptBinding) error { return nil })); err != nil {
					t.Fatal("unchanged recovery intent cannot bind", err)
				}
				if err = restarted.RecordStartSubmission(ctx, start); err == nil {
					t.Fatal("old start accepted after Bind")
				}
				if _, err = restarted.AcceptSubmission(ctx, submit); err == nil {
					t.Fatal("old submit accepted after Bind")
				}
				start.AttemptID, start.WorkloadGeneration = binding.AttemptID, binding.WorkloadGeneration
				submit.AttemptID, submit.WorkloadGeneration = binding.AttemptID, binding.WorkloadGeneration
				if err = restarted.RecordStartSubmission(ctx, start); err != nil {
					t.Fatal("new attempt blocked before Finish", err)
				}
				receipt, err := restarted.AcceptSubmission(ctx, submit)
				if err != nil {
					t.Fatal("new attempt submit blocked before Finish", err)
				}
				if err = restarted.RecordStartSubmission(ctx, start); err != nil {
					t.Fatal("new start replay failed", err)
				}
				if again, err := restarted.AcceptSubmission(ctx, submit); err != nil || again != receipt {
					t.Fatal("new submit replay changed receipt", err)
				}
				if err = restarted.FinishLearnerRecovery(ctx, id, intent.Attempt); err != nil {
					t.Fatal(err)
				}
				s = h.state(t)
				if len(s.Starts) != 2 || len(s.Updates) != 2 || len(s.Attempts) != 1 || s.Recovery != nil || !reflect.DeepEqual(fenced.Assignments, s.Assignments) || !reflect.DeepEqual(fenced.Plan.Slots, s.Plan.Slots) {
					t.Fatal("recovery lost, duplicated, or reassigned work")
				}
			})
		}
	}
}
