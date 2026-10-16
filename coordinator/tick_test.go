package coordinator

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/run"
)

type forbiddenEvaluator struct{ t *testing.T }

func (e forbiddenEvaluator) Evaluate(context.Context, run.EvaluationAssignment) (run.EvaluationResult, error) {
	e.t.Fatal("service tick invoked numerical evaluator")
	return run.EvaluationResult{}, errors.New("unexpected evaluation")
}

func TestServiceTickWaitsForBoundEvaluation(t *testing.T) {
	h := setup(t)
	ctx := context.Background()
	a := h.start(t)
	before := h.state(t)
	if err := h.m.Tick(ctx, h.plan.Cohort.RunID, 100); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, h.state(t)) {
		t.Fatal("pending learners caused a transition")
	}
	h.acceptAll(t, a)
	for _, phase := range []Phase{Aggregating, Publishing, Evaluating} {
		if err := h.m.Tick(ctx, h.plan.Cohort.RunID, 100); err != nil {
			t.Fatal(err)
		}
		if h.state(t).Phase != phase {
			t.Fatal("wrong transition", h.state(t).Phase, phase)
		}
	}
	h.m.Evaluator = forbiddenEvaluator{t}
	before = h.state(t)
	if err := h.m.Tick(ctx, h.plan.Cohort.RunID, 100); err != nil || !reflect.DeepEqual(before, h.state(t)) {
		t.Fatal("pending evaluation changed state", err)
	}
	h.clock.now.Store(before.Evaluation.ExpiresAt)
	if err := h.m.Tick(ctx, h.plan.Cohort.RunID, 100); err != nil {
		t.Fatal(err)
	}
	s := h.state(t)
	if s.Phase != Failed || s.Failure != "global evaluation deadline expired" || len(h.repo.results) != 0 {
		t.Fatal("failed evaluation not durably reported", s)
	}
	if err := h.m.Tick(ctx, h.plan.Cohort.RunID, 100); err != nil || !reflect.DeepEqual(s, h.state(t)) {
		t.Fatal("terminal tick changed state", err)
	}
}

func TestServiceTickPreservesDeadlineAfterRestart(t *testing.T) {
	h := setup(t)
	a := h.start(t)
	restarted := h.m
	h.clock.now.Store(a[0].ExpiresAt)
	if err := restarted.Tick(context.Background(), h.plan.Cohort.RunID, 100); err != nil {
		t.Fatal(err)
	}
	s := h.state(t)
	if s.Phase != Failed || s.Failure != "learner round deadline expired" || s.Assignments[0] != a[0] {
		t.Fatal("restart extended or replaced round", s)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := restarted.Tick(ctx, h.plan.Cohort.RunID, 100); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
