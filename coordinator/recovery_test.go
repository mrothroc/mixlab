package coordinator

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
)

// Hold all callers after reading the same revision, before any can commit.
type barrierRepo struct {
	Repository
	ready sync.WaitGroup
}

func (r *barrierRepo) Load(ctx context.Context, id string) (State, error) {
	s, e := r.Repository.Load(ctx, id)
	r.ready.Done()
	r.ready.Wait()
	return s, e
}

func concurrentCalls(t *testing.T, h *harness, f func(Model) error) int {
	t.Helper()
	const count = 8
	r := &barrierRepo{Repository: h.repo}
	r.ready.Add(count)
	m := h.m
	m.Repository = r
	results := make(chan error, count)
	for range count {
		go func() { results <- f(m) }()
	}
	success := 0
	for range count {
		e := <-results
		if e == nil {
			success++
		} else if !errors.Is(e, ErrConflict) {
			t.Errorf("unexpected concurrent result: %v", e)
		}
	}
	return success
}

func TestConcurrentAcceptanceAndAdvancement(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	ctx := context.Background()
	id := h.plan.Cohort.RunID
	for i := range a {
		u := h.update(t, a[i], i)
		before := h.state(t).Revision
		if concurrentCalls(t, h, func(m Model) error { _, e := m.Accept(ctx, id, u); return e }) == 0 {
			t.Fatal("no accepted update")
		}
		if h.state(t).Revision != before+1 || len(h.repo.receipts) != i+1 {
			t.Fatal("duplicate acceptance")
		}
	}
	for _, phase := range []Phase{Aggregating, Publishing, Evaluating, Ready} {
		before := h.state(t).Revision
		if n := concurrentCalls(t, h, func(m Model) error { return m.Advance(ctx, id) }); n != 1 {
			t.Fatalf("%s: %d commits", phase, n)
		}
		s := h.state(t)
		if s.Phase != phase || s.Revision != before+1 {
			t.Fatal("concurrent calls advanced more than one boundary")
		}
	}
	if h.state(t).Round != 1 || len(h.repo.rounds) != 1 || len(h.repo.results) != 1 || len(h.repo.decisions) != 1 {
		t.Fatal("round/evaluation published more than once")
	}
}

func TestRestoredStateRejectsMalformedBindings(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	h.acceptAll(t, a)
	for range 3 {
		h.advance(t)
	}
	good := h.state(t)
	for name, mutate := range map[string]func(*State){
		"revision":    func(s *State) { s.Revision = 0 },
		"commit":      func(s *State) { s.Commit = nil },
		"assignments": func(s *State) { s.Assignments = nil },
		"order":       func(s *State) { s.Assignments[0], s.Assignments[1] = s.Assignments[1], s.Assignments[0] },
		"duplicate":   func(s *State) { s.Updates[1] = s.Updates[0] },
		"partial":     func(s *State) { s.Updates = s.Updates[:1] },
		"candidate":   func(s *State) { s.Candidate = nil },
		"evaluation":  func(s *State) { s.Evaluation = nil },
		"policy":      func(s *State) { s.Evaluation.Batches++ },
		"private":     func(s *State) { s.PrivateHashes[0] = s.Plan.SHA256 },
		"phase":       func(s *State) { s.Phase = "unknown" },
	} {
		t.Run(name, func(t *testing.T) {
			bad := copyValue(good)
			mutate(&bad)
			h.repo.state = &bad
			if e := h.m.Advance(context.Background(), h.plan.Cohort.RunID); e == nil {
				t.Fatal("restored corrupt state advanced")
			}
			if !reflect.DeepEqual(bad, h.state(t)) {
				t.Fatal("rejection changed state")
			}
		})
	}
}

func TestStaleUnseenUpdateAndTokenOverflow(t *testing.T) {
	ctx := context.Background()
	t.Run("overflow", func(t *testing.T) {
		h := setup(t)
		a := h.start(t)
		u := h.update(t, a[0], 0)
		u.EffectiveTokens = ^uint64(0)
		if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u); e != nil {
			t.Fatal(e)
		}
		before := h.state(t)
		if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, h.update(t, a[1], 1)); e == nil {
			t.Fatal("overflow accepted")
		}
		if !reflect.DeepEqual(before, h.state(t)) {
			t.Fatal("overflow changed state")
		}
	})
	t.Run("stale", func(t *testing.T) {
		h := setup(t)
		a := h.start(t)
		u := h.acceptAll(t, a)
		for range 4 {
			h.advance(t)
		}
		if _, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100); e != nil {
			t.Fatal(e)
		}
		u[0].ID = "never-accepted"
		before := h.state(t)
		if _, e := h.m.Accept(ctx, h.plan.Cohort.RunID, u[0]); e == nil {
			t.Fatal("unseen stale update accepted")
		}
		if !reflect.DeepEqual(before, h.state(t)) {
			t.Fatal("stale update changed state")
		}
	})
}

func TestEvaluationDeadlineAndWrongModel(t *testing.T) {
	for _, expired := range []bool{false, true} {
		h := setup(t)
		h.acceptAll(t, h.start(t))
		for range 3 {
			h.advance(t)
		}
		before := h.state(t)
		if expired {
			h.clock.now.Store(before.Evaluation.ExpiresAt)
		} else {
			h.eval.wrongModel = true
		}
		if e := h.m.Advance(context.Background(), h.plan.Cohort.RunID); e == nil {
			t.Fatal("bad evaluation advanced")
		}
		if !reflect.DeepEqual(before, h.state(t)) || len(h.repo.results) != 0 {
			t.Fatal("bad evaluation changed state")
		}
	}
}

func TestConcurrentConflictingUpdateIDs(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	u := h.update(t, a[0], 0)
	const count = 2
	r := &barrierRepo{Repository: h.repo}
	r.ready.Add(count)
	m := h.m
	m.Repository = r
	results := make(chan error, count)
	for i := range count {
		v := u
		v.Loss += float64(i)
		go func(v learner.Update) {
			_, e := m.Accept(context.Background(), h.plan.Cohort.RunID, v)
			results <- e
		}(v)
	}
	success := 0
	for range count {
		if e := <-results; e == nil {
			success++
		} else if !errors.Is(e, ErrConflict) {
			t.Fatal(e)
		}
	}
	if success != 1 || len(h.state(t).Updates) != 1 || len(h.repo.receipts) != 1 {
		t.Fatal("conflicting semantic update accepted twice")
	}
}

func TestNumericalOverflowNeverPublishes(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	ctx := context.Background()
	for i := range a {
		u := h.update(t, a[i], i)
		ref, e := h.artifacts.Put(ctx, "delta_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{math.MaxFloat32}}})
		if e != nil {
			t.Fatal(e)
		}
		u.Delta, u.DeltaL2, u.DeltaMaxAbs = ref, float64(float32(math.MaxFloat32)), float64(float32(math.MaxFloat32))
		if _, e = h.m.Accept(ctx, h.plan.Cohort.RunID, u); e != nil {
			t.Fatal(e)
		}
	}
	h.advance(t)
	before, puts := h.state(t), h.artifacts.puts
	// LR=1, beta=.5 produces a 1.5*MaxFloat32 step from finite deltas.
	if e := h.m.Advance(ctx, h.plan.Cohort.RunID); e == nil {
		t.Fatal("overflow published")
	}
	if !reflect.DeepEqual(before, h.state(t)) || h.artifacts.puts != puts || len(h.repo.rounds) != 0 {
		t.Fatal("overflow changed global state/artifacts")
	}
}

func TestInvalidEvaluationPolicyRejectsBeforeStaging(t *testing.T) {
	for _, split := range []string{"", "has space", "\n", "\xff"} {
		h := setup(t)
		h.policy.Split = split
		if e := h.m.Stage(context.Background(), h.plan, h.policy); e == nil || h.repo.state != nil {
			t.Fatal("invalid evaluation policy staged")
		}
	}
}
