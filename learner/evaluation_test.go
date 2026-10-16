package learner

import (
	"context"
	"io"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

func (p *fakePeer) ReadEvaluation(_ context.Context, _ run.EvaluationAssignment, w io.Writer) error {
	_, err := w.Write(p.base)
	return err
}
func TestRuntimeEvaluationPersistsResultWithoutTrainingMutation(t *testing.T) {
	f := setup(t)
	f.run(t)
	_, j, err := f.r.Store.load()
	if err != nil {
		t.Fatal(err)
	}
	a := run.EvaluationAssignment{Format: "mixlab_evaluation_assignment_v1", ID: "eval0", Binding: f.a.Binding, Coordinator: f.a.Coordinator, Round: 0, Model: f.a.BaseModel, ValidationHash: f.a.Contract.DatasetHash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1, WorkerID: f.a.WorkerID, Precision: j.Completed.Precision, PrecisionHash: f.a.PrecisionHash, IssuedAt: 1200, ExpiresAt: 1800}
	calls := 0
	evaluate := func(_ context.Context, _ run.EvaluationAssignment, m []optimizer.Tensor) (float64, uint64, error) {
		calls++
		m[0].Values[0] = -100
		return 2, 8, nil
	}
	first, err := f.r.Evaluate(context.Background(), a, evaluate)
	if err != nil {
		t.Fatal(err)
	}
	again, err := f.r.Evaluate(context.Background(), a, evaluate)
	if err != nil || first != again || calls != 1 {
		t.Fatal("evaluation repeated", calls, err)
	}
	_, after, err := f.r.Store.load()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(j.Completed, after.Completed) || !reflect.DeepEqual(j.Update, after.Update) || f.calls != 1 {
		t.Fatal("evaluation mutated private state")
	}
	a.Batches++
	if _, err = f.r.Evaluate(context.Background(), a, evaluate); err == nil {
		t.Fatal("evaluation policy changed")
	}
}
