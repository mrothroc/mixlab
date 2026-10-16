package train

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/workerjob"
)

type observedLocalEvaluation struct {
	*fakeLocalLearner
	seen   *[][]int
	closed *bool
	cancel context.CancelFunc
	loss   *float32
	err    error
}

func (e *observedLocalEvaluation) evaluate(b objectiveBatch) (float32, error) {
	*e.seen = append(*e.seen, append([]int(nil), b.x...))
	if e.cancel != nil {
		e.cancel()
	}
	if e.err != nil {
		return 0, e.err
	}
	if e.loss != nil {
		return *e.loss, nil
	}
	return float32(len(*e.seen)), nil
}

func TestLocalLearnerEvaluationRejectsModelBeforeEngine(t *testing.T) {
	cfg, path, _, model := learnerFixture(t, 2)
	hash, err := data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	v := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1}
	for name, mutate := range map[string]func([]LocalLearnerTensor) []LocalLearnerTensor{
		"count":  func(m []LocalLearnerTensor) []LocalLearnerTensor { return m[:len(m)-1] },
		"name":   func(m []LocalLearnerTensor) []LocalLearnerTensor { m[0].Name = "wrong"; return m },
		"order":  func(m []LocalLearnerTensor) []LocalLearnerTensor { m[0], m[1] = m[1], m[0]; return m },
		"shape":  func(m []LocalLearnerTensor) []LocalLearnerTensor { m[0].Shape[0]++; return m },
		"length": func(m []LocalLearnerTensor) []LocalLearnerTensor { m[0].Data = m[0].Data[:1]; return m },
		"nan":    func(m []LocalLearnerTensor) []LocalLearnerTensor { m[len(m)-1].Data[0] = float32(math.NaN()); return m },
		"inf":    func(m []LocalLearnerTensor) []LocalLearnerTensor { m[0].Data[0] = float32(math.Inf(1)); return m },
	} {
		t.Run(name, func(t *testing.T) {
			loss, tokens, err := evaluateLocalLearnerModel(context.Background(), cfg, mutate(cloneLearnerValue(model)), v, precision.Policy{}, func([][]float32, bool) (localLearnerEngine, error) {
				t.Fatal("malformed model reached engine")
				return nil, nil
			})
			if err == nil || loss != 0 || tokens != 0 {
				t.Fatal("malformed model produced metrics", err)
			}
		})
	}
}

func TestLocalLearnerEvaluationClosesOnFailedMetrics(t *testing.T) {
	cfg, path, _, model := learnerFixture(t, 2)
	hash, err := data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	v := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 3}
	for _, bad := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		closed := false
		var seen [][]int
		loss, tokens, err := evaluateLocalLearnerModel(context.Background(), cfg, model, v, precision.Policy{}, func(w [][]float32, _ bool) (localLearnerEngine, error) {
			return &observedLocalEvaluation{fakeLocalLearner: &fakeLocalLearner{w: w}, seen: &seen, closed: &closed, loss: &bad}, nil
		})
		if err == nil || loss != 0 || tokens != 0 || !closed || len(seen) != 1 {
			t.Fatalf("invalid loss %v leaked metrics/resources: %v", bad, err)
		}
	}
	want := errors.New("evaluation engine failed")
	closed := false
	var seen [][]int
	_, _, err = evaluateLocalLearnerModel(context.Background(), cfg, model, v, precision.Policy{}, func(w [][]float32, _ bool) (localLearnerEngine, error) {
		return &observedLocalEvaluation{fakeLocalLearner: &fakeLocalLearner{w: w}, seen: &seen, closed: &closed, err: want}, nil
	})
	if !errors.Is(err, want) || !closed {
		t.Fatal("engine failure lost or engine leaked", err)
	}
}

func TestLocalLearnerEvaluationBatchMaskAccounting(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	cfg.Training.DatasetRecordFraming = true
	b := data.Batch{X: make([]int, 8), Y: make([]int, 8), SeqLen: 4, LossMask: []float32{1, 1, 0, 0, 1, 0, 0, 0}}
	loss, tokens, err := evaluateLocalBatch(cfg, &fakeLocalLearner{}, b, 0)
	if err != nil || loss != 2 || tokens != 3 {
		t.Fatal("masked denominator", loss, tokens, err)
	}
	for _, invalid := range []float32{-1, .5, 2, float32(math.NaN()), float32(math.Inf(1))} {
		b.LossMask[0] = invalid
		if _, _, err := evaluateLocalBatch(cfg, &fakeLocalLearner{}, b, 0); err == nil {
			t.Fatalf("invalid mask %v accepted", invalid)
		}
	}
}
func (e *observedLocalEvaluation) close() { *e.closed = true }

func TestLocalLearnerStreamingGlobalEvaluation(t *testing.T) {
	cfg, path, _, model := learnerFixture(t, 2)
	hash, err := data.DistributedDatasetIdentity(path)
	if err != nil {
		t.Fatal(err)
	}
	v := workerjob.LearnerValidation{Pattern: path, Hash: hash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 3}
	policy := precision.Policy{ComputeDType: "float32", Backend: "metal"}
	before, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var previous [][]int
	for range 2 {
		var seen [][]int
		closed := false
		factory := func(w [][]float32, eval bool) (localLearnerEngine, error) {
			if !eval || !reflect.DeepEqual(w[0], model[0].Data) {
				t.Fatal("wrong eval engine or global model")
			}
			return &observedLocalEvaluation{fakeLocalLearner: &fakeLocalLearner{w: w}, seen: &seen, closed: &closed}, nil
		}
		loss, tokens, err := evaluateLocalLearnerModel(context.Background(), cfg, model, v, policy, factory)
		if err != nil {
			t.Fatal(err)
		}
		if loss != 2 || tokens != 24 || len(seen) != 3 || !closed {
			t.Fatalf("loss=%v tokens=%d seen=%d closed=%v", loss, tokens, len(seen), closed)
		}
		if previous != nil && !reflect.DeepEqual(previous, seen) {
			t.Fatal("validation replay changed")
		}
		previous = seen
	}
	after, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("evaluation mutated caller config")
	}
	for _, mutate := range []func(*workerjob.LearnerValidation){
		func(v *workerjob.LearnerValidation) { v.Hash = strings.Repeat("f", 64) },
		func(v *workerjob.LearnerValidation) { v.BatchTokens = 16 },
		func(v *workerjob.LearnerValidation) { v.SequenceLength = 2 },
		func(v *workerjob.LearnerValidation) { v.Batches = 65537 },
	} {
		bad := v
		mutate(&bad)
		_, _, err := evaluateLocalLearnerModel(context.Background(), cfg, model, bad, policy, func([][]float32, bool) (localLearnerEngine, error) {
			t.Fatal("invalid policy reached engine")
			return nil, nil
		})
		if err == nil {
			t.Fatal("invalid validation policy accepted")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	closed := false
	var seen [][]int
	_, _, err = evaluateLocalLearnerModel(ctx, cfg, model, v, policy, func(w [][]float32, _ bool) (localLearnerEngine, error) {
		return &observedLocalEvaluation{fakeLocalLearner: &fakeLocalLearner{w: w}, seen: &seen, closed: &closed, cancel: cancel}, nil
	})
	if !errors.Is(err, context.Canceled) || !closed || len(seen) != 1 {
		t.Fatalf("cancellation not bounded: %v", err)
	}
}
