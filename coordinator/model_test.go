package coordinator

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
)

func TestCommittedCohortMultiRoundOracle(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	a := h.start(t)
	for round, want := range []float32{2.03125, 1.0263671875, .732208251953125} {
		u := h.acceptAll(t, a)
		for _, phase := range []Phase{Aggregating, Publishing, Evaluating, Ready} {
			h.advance(t)
			s := h.state(t)
			if phase == Ready && round == 2 {
				phase = Complete
			}
			if s.Phase != phase {
				t.Fatalf("got %s want %s", s.Phase, phase)
			}
		}
		s := h.state(t)
		ts, e := h.artifacts.Load(ctx, s.Base)
		if e != nil {
			t.Fatal(e)
		}
		if math.Abs(float64(ts[0].Values[0]-want)) > 1e-6 {
			t.Fatalf("round %d: got %v want %v", round, ts, want)
		}
		if s.Round != uint64(round+1) || len(h.repo.rounds) != round+1 || len(h.repo.results) != round+1 {
			t.Fatal("incorrect advancement")
		}
		for _, old := range u {
			if _, e = h.m.Accept(ctx, h.plan.Cohort.RunID, old); e != nil {
				t.Fatal("lost ACK retry after advancement", e)
			}
		}
		if round < 2 {
			h.clock.now.Add(10)
			a, e = h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100)
			if e != nil {
				t.Fatal(e)
			}
			for i, v := range a {
				if v.PreviousPrivateHash != u[i].PrivateHash || v.BaseModel != s.Base {
					t.Fatal("successor assignment lost state binding")
				}
			}
		}
	}
	if _, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e == nil {
		t.Fatal("opened beyond configured steps")
	}
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); e != nil {
		t.Fatal(e)
	}
}

func TestStagedPlanNeverPermitsRound(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	if e := h.m.Stage(ctx, h.plan, h.policy); e != nil {
		t.Fatal(e)
	}
	if _, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e == nil {
		t.Fatal("staged plan trained")
	}
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); e == nil {
		t.Fatal("staged plan advanced")
	}
	h.clock.now.Store(h.plan.ExpiresAt)
	if e := h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); e == nil {
		t.Fatal("expired plan committed")
	}
	if h.state(t).Phase != Staged {
		t.Fatal("failed commit mutated state")
	}
}

func TestUpdateAdmissionFailures(t *testing.T) {
	for name, change := range map[string]func(*learner.Update){
		"stale": func(u *learner.Update) { u.Round++ }, "skipped": func(u *learner.Update) { u.Skipped = 1 }, "extra_work": func(u *learner.Update) { u.Attempted++; u.Committed++ }, "zero_count": func(u *learner.Update) { u.EffectiveTokens = 0 },
		"participant": func(u *learner.Update) { u.ParticipantID = "other" }, "worker": func(u *learner.Update) { u.WorkerID = "other" }, "island": func(u *learner.Update) { u.IslandID = "other" }, "policy": func(u *learner.Update) { u.PrecisionHash = "" }, "missing_private": func(u *learner.Update) { u.PrivateHash = "" },
		"norm": func(u *learner.Update) { u.DeltaL2++ }, "tensor_count": func(u *learner.Update) { u.TensorCount++ }, "size": func(u *learner.Update) { u.Delta.Ref.Bytes++ }, "digest": func(u *learner.Update) { u.Delta.Ref.SHA256 = u.PrivateHash }, "dtype": func(u *learner.Update) { u.DType = "bf16" }, "nonfinite_summary": func(u *learner.Update) { u.DeltaMaxAbs = math.Inf(1) },
	} {
		t.Run(name, func(t *testing.T) {
			h := setup(t)
			a := h.start(t)
			u := h.update(t, a[0], 0)
			before := h.state(t)
			change(&u)
			if _, e := h.m.Accept(context.Background(), h.plan.Cohort.RunID, u); e == nil {
				t.Fatal("accepted bad update")
			}
			if !reflect.DeepEqual(before, h.state(t)) || len(h.repo.receipts) != 0 {
				t.Fatal("rejection mutated state")
			}
		})
	}
}

func TestDuplicateAndConflictingUpdates(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	a := h.start(t)
	u := h.update(t, a[0], 0)
	first, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u)
	if e != nil {
		t.Fatal(e)
	}
	before := h.state(t)
	again, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u)
	if e != nil || again != first || !reflect.DeepEqual(before, h.state(t)) {
		t.Fatal("retry not idempotent", e)
	}
	changed := u
	changed.Loss++
	if _, e = h.m.Accept(ctx, h.plan.Cohort.RunID, changed); !errors.Is(e, ErrConflict) {
		t.Fatal("same ID accepted different content", e)
	}
	changed = u
	changed.ID = "another"
	if _, e = h.m.Accept(ctx, h.plan.Cohort.RunID, changed); !errors.Is(e, ErrConflict) {
		t.Fatal("second update from worker accepted", e)
	}
	if e = h.m.Advance(ctx, h.plan.Cohort.RunID); e == nil {
		t.Fatal("partial cohort aggregated")
	}
	negative, _ := json.Marshal(u)
	var fields map[string]json.RawMessage
	if e = json.Unmarshal(negative, &fields); e != nil {
		t.Fatal(e)
	}
	fields["effective_tokens"] = json.RawMessage(`-1`)
	negative, _ = json.Marshal(fields)
	var decoded learner.Update
	if e = json.Unmarshal(negative, &decoded); e == nil {
		t.Fatal("negative count accepted")
	}
}

func TestLostAcknowledgmentAndPublicationRecovery(t *testing.T) {
	for _, after := range []bool{false, true} {
		t.Run(map[bool]string{false: "before_write", true: "after_write"}[after], func(t *testing.T) {
			h := setup(t)
			ctx := context.Background()
			a := h.start(t)
			u := h.update(t, a[0], 0)
			h.repo.failBefore = !after
			h.repo.failAfter = after
			if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u); !errors.Is(e, errInjected) {
				t.Fatal(e)
			}
			// A new coordinator object has no private in-memory progress to recover.
			h.m = Model{Repository: h.repo, Artifacts: h.artifacts, Allocations: h.alloc, Clock: h.clock, Evaluator: h.eval}
			if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u); e != nil {
				t.Fatal(e)
			}
			if len(h.state(t).Updates) != 1 || len(h.repo.receipts) != 1 {
				t.Fatal("duplicate acceptance")
			}
			u1 := h.update(t, a[1], 1)
			if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u1); e != nil {
				t.Fatal(e)
			}
			for _, phase := range []Phase{Open, Aggregating, Publishing, Evaluating} {
				if h.state(t).Phase != phase {
					t.Fatal("unexpected phase")
				}
				before := h.state(t)
				h.repo.failBefore = !after
				h.repo.failAfter = after
				if e := h.m.Advance(ctx, h.plan.Cohort.RunID); !errors.Is(e, errInjected) {
					t.Fatal(e)
				}
				if !after && !reflect.DeepEqual(before, h.state(t)) {
					t.Fatal("failed write changed state")
				}
				// Recovery uses the stored phase. Re-running aggregation before its
				// CAS only creates the same immutable blobs; it never changes theta.
				if !after {
					h.advance(t)
				}
			}
			if h.state(t).Round != 1 || len(h.repo.rounds) != 1 || len(h.repo.results) != 1 {
				t.Fatal("round advanced more than once")
			}
		})
	}
}

func TestArtifactAndEvaluationFailures(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	a := h.start(t)
	h.acceptAll(t, a)
	h.advance(t)
	before := h.state(t)
	h.artifacts.failPut = h.artifacts.puts + 2 // model staged, momentum write fails
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); !errors.Is(e, errInjected) {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(before, h.state(t)) {
		t.Fatal("partial tensor publication changed state")
	}
	h.artifacts.failPut = 0
	h.advance(t)
	h.advance(t)
	published := h.state(t)
	h.eval.fail = true
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); !errors.Is(e, errInjected) {
		t.Fatal(e)
	}
	if _, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e == nil {
		t.Fatal("training before global evaluation")
	}
	if !reflect.DeepEqual(published, h.state(t)) {
		t.Fatal("failed eval erased global checkpoint")
	}
	h.eval.fail = false
	h.eval.mutatePrivate = true
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); e == nil {
		t.Fatal("evaluation changed private state")
	}
	h.eval.mutatePrivate = false
	h.eval.cache = map[string]run.EvaluationResult{}
	h.advance(t)
}

func TestAllocationAndDeadlineGates(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	if e := h.m.Stage(ctx, h.plan, h.policy); e != nil {
		t.Fatal(e)
	}
	if e := h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); e != nil {
		t.Fatal(e)
	}
	h.alloc.fail = true
	if _, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e == nil {
		t.Fatal("opened with lost allocation")
	}
	h.alloc.fail = false
	a, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100)
	if e != nil {
		t.Fatal(e)
	}
	again, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 200)
	if e != nil || !reflect.DeepEqual(a, again) {
		t.Fatal("assignment changed on retry")
	}
	u := h.update(t, a[0], 0)
	h.clock.now.Store(a[0].ExpiresAt)
	if _, e = h.m.Accept(ctx, h.plan.Cohort.RunID, u); e == nil {
		t.Fatal("expired update accepted")
	}
	if e = h.m.Fail(ctx, h.plan.Cohort.RunID, "allocation expired"); e != nil {
		t.Fatal(e)
	}
	if _, e = h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e == nil {
		t.Fatal("failed run restarted")
	}
}

func TestEarlyStopDecisionIsAtomic(t *testing.T) {
	h := setup(t)
	target := 2.1
	h.policy.TargetLoss = &target
	a := h.start(t)
	h.acceptAll(t, a)
	for range 4 {
		h.advance(t)
	}
	s := h.state(t)
	if s.Phase != Complete || s.Round != 1 || h.repo.decisions[0].Decision != "complete" {
		t.Fatal("target decision not committed")
	}
}
