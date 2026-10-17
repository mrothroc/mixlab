package coordinator

import (
	"context"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

func TestLearnerRecoveryPhasesAndDurableLimit(t *testing.T) {
	for _, phase := range []Phase{Open, Evaluating, Ready} {
		t.Run(string(phase), func(t *testing.T) {
			h := setup(t)
			ctx := context.Background()
			a := h.start(t)
			ah, _ := run.Digest(a[0])
			if err := h.m.RecordStart(ctx, learner.Start{Format: "mixlab_learner_start_v1", Binding: a[0].Binding, WorkerID: a[0].WorkerID, AssignmentHash: ah, PrivateHash: h.plan.Contract.DatasetHash}); err != nil {
				t.Fatal(err)
			}
			if phase != Open {
				h.acceptAll(t, a)
				for h.state(t).Phase != phase {
					h.advance(t)
				}
			}
			before := h.state(t)
			for retry := 0; retry < LearnerRestartLimit; retry++ {
				if err := h.m.BeginLearnerRecovery(ctx, h.plan.Cohort.RunID, a[0].WorkerID); err != nil {
					t.Fatal(err)
				}
				intent := *h.state(t).Recovery
				expected := h.plan.Contract.DatasetHash
				if phase != Open {
					expected = h.plan.Contract.ConfigHash
				}
				if intent.PrivateHash != expected {
					t.Fatal("wrong private head", intent.PrivateHash, expected)
				}
				restarted := h.m
				h.clock.now.Add(1)
				if err := restarted.BeginLearnerRecovery(ctx, h.plan.Cohort.RunID, a[0].WorkerID); err != nil {
					t.Fatal(err)
				}
				if *h.state(t).Recovery != intent {
					t.Fatal("restart replaced recovery intent/deadline")
				}
				reg := h.commit.Registrations[0]
				_, generation, _ := h.state(t).CurrentAttempt(a[0].WorkerID)
				binding := run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: h.commit.Binding, WorkerID: a[0].WorkerID, PreviousAttemptID: intent.Previous, AttemptID: intent.Attempt, WorkloadGeneration: generation + 1, Allocation: reg.Slot.Allocation, NodeAuthorizationHash: h.plan.SHA256, PrivateHash: intent.PrivateHash, PrecisionHash: reg.PrecisionHash}
				auth := recoveryCheck(func(context.Context, run.AttemptBinding) error { return nil })
				for range 2 {
					if err := restarted.BindAttempt(ctx, binding, auth); err != nil {
						t.Fatal(err)
					}
				}
				if err := restarted.FinishLearnerRecovery(ctx, h.plan.Cohort.RunID, intent.Attempt); err != nil {
					t.Fatal(err)
				}
			}
			s := h.state(t)
			if len(s.Attempts) != LearnerRestartLimit || !reflect.DeepEqual(before.Assignments, s.Assignments) || !reflect.DeepEqual(before.Evaluation, s.Evaluation) || !reflect.DeepEqual(before.Plan.Slots, s.Plan.Slots) {
				t.Fatal("recovery changed semantic work")
			}
			if err := h.m.BeginLearnerRecovery(ctx, h.plan.Cohort.RunID, a[0].WorkerID); err != nil {
				t.Fatal(err)
			}
			if s = h.state(t); s.Phase != Failed || s.Failure == "" {
				t.Fatal("restart limit not terminal", s.Phase)
			}
		})
	}
}

func TestLearnerRecoveryMissingPrivateFailsImmediately(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	if err := h.m.BeginLearnerRecovery(context.Background(), h.plan.Cohort.RunID, a[0].WorkerID); err != nil {
		t.Fatal(err)
	}
	s := h.state(t)
	if s.Phase != Failed || s.Recovery != nil || len(s.Attempts) != 0 {
		t.Fatal("missing private state initiated recovery")
	}
}

func TestLearnerStatusOutageBudgetSurvivesRestart(t *testing.T) {
	h := setup(t)
	h.start(t)
	ctx := context.Background()
	id := h.plan.Cohort.RunID
	if err := h.m.ObserveLearnerAvailability(ctx, id, true); err != nil {
		t.Fatal(err)
	}
	start := h.state(t).LearnerUnavailableAt
	// A slow management request is not a stopped child and spends no attempt.
	h.clock.now.Add(120)
	restarted := h.m
	if err := restarted.ObserveLearnerAvailability(ctx, id, true); err != nil {
		t.Fatal(err)
	}
	if s := h.state(t); s.Phase == Failed || s.Recovery != nil || len(s.Attempts) != 0 || s.LearnerUnavailableAt != start {
		t.Fatal("slowness triggered recovery or reset outage", s)
	}
	h.clock.now.Add(LearnerRecoverySeconds - 120)
	if err := restarted.ObserveLearnerAvailability(ctx, id, true); err != nil {
		t.Fatal(err)
	}
	if s := h.state(t); s.Phase != Failed || s.LearnerUnavailableAt != start {
		t.Fatal("outage budget reset", s)
	}
}
