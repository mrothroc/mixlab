package coordinator

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

var errInjected = errors.New("injected failure")

func copyValue[T any](v T) T {
	b, e := json.Marshal(v)
	if e != nil {
		panic(e)
	}
	var out T
	if e = json.Unmarshal(b, &out); e != nil {
		panic(e)
	}
	return out
}

type memoryRepo struct {
	mu                    sync.Mutex
	state                 *State
	receipts              map[string]learner.Receipt
	rounds                map[uint64]run.Round
	results               map[uint64]run.EvaluationResult
	decisions             map[uint64]run.EvaluationDecision
	failBefore, failAfter bool
}

func (r *memoryRepo) Load(_ context.Context, id string) (State, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.state == nil || r.state.Plan.Cohort.RunID != id {
		return State{}, ErrNotFound
	}
	return copyValue(*r.state), nil
}
func (r *memoryRepo) Receipt(_ context.Context, id, update string) (learner.Receipt, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	v, ok := r.receipts[id+"/"+update]
	if !ok {
		return v, ErrNotFound
	}
	return v, nil
}
func (r *memoryRepo) Save(_ context.Context, id string, revision uint64, c Change) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.failBefore {
		r.failBefore = false
		return errInjected
	}
	if (r.state == nil && revision != 0) || (r.state != nil && (r.state.Revision != revision || r.state.Plan.Cohort.RunID != id)) {
		return ErrConflict
	}
	if c.Next.Revision != revision+1 {
		return fmt.Errorf("bad revision")
	}
	if c.Receipt != nil {
		key := id + "/" + c.Receipt.UpdateID
		if old, ok := r.receipts[key]; ok && old != *c.Receipt {
			return ErrConflict
		}
		r.receipts[key] = *c.Receipt
	}
	if c.Published != nil {
		if _, ok := r.rounds[c.Published.Round]; ok {
			return fmt.Errorf("round published twice")
		}
		r.rounds[c.Published.Round] = copyValue(*c.Published)
	}
	if c.Evaluation != nil {
		if _, ok := r.results[c.Evaluation.Assignment.Round]; ok {
			return fmt.Errorf("evaluation published twice")
		}
		r.results[c.Evaluation.Assignment.Round] = copyValue(*c.Evaluation)
		r.decisions[c.Decision.Round] = *c.Decision
	}
	v := copyValue(c.Next)
	r.state = &v
	if r.failAfter {
		r.failAfter = false
		return errInjected
	}
	return nil
}

type memoryArtifacts struct {
	mu       sync.Mutex
	blobs    map[string][]byte
	failPut  int
	puts     int
	failLoad bool
}

func (a *memoryArtifacts) Put(_ context.Context, kind string, ts []optimizer.Tensor) (run.Blob, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.puts++
	if a.failPut == a.puts {
		return run.Blob{}, errInjected
	}
	if e := optimizer.Validate(ts); e != nil {
		return run.Blob{}, e
	}
	b, e := json.Marshal(ts)
	if e != nil {
		return run.Blob{}, e
	}
	h := sha256.Sum256(b)
	ref := run.Blob{Ref: artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(b))}, Kind: kind}
	if e = ref.Validate(); e != nil {
		return ref, e
	}
	a.blobs[ref.Ref.SHA256] = append([]byte(nil), b...)
	return ref, nil
}
func (a *memoryArtifacts) Load(_ context.Context, ref run.Blob) ([]optimizer.Tensor, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.failLoad {
		return nil, errInjected
	}
	if e := ref.Validate(); e != nil {
		return nil, e
	}
	b, ok := a.blobs[ref.Ref.SHA256]
	if !ok {
		return nil, ErrNotFound
	}
	h := sha256.Sum256(b)
	if uint64(len(b)) != ref.Ref.Bytes || hex.EncodeToString(h[:]) != ref.Ref.SHA256 {
		return nil, fmt.Errorf("artifact bytes/digest mismatch")
	}
	var ts []optimizer.Tensor
	e := json.Unmarshal(b, &ts)
	return ts, e
}

type fakeClock struct{ now atomic.Int64 }

func (c *fakeClock) Unix() int64 { return c.now.Load() }

type fakeAllocations struct {
	fail  bool
	calls atomic.Int64
}

func (a *fakeAllocations) Ready(context.Context, run.Slot) error {
	a.calls.Add(1)
	if a.fail {
		return errInjected
	}
	return nil
}

type fakeEvaluator struct {
	mu                              sync.Mutex
	repo                            *memoryRepo
	clock                           *fakeClock
	calls                           int
	loss                            float64
	fail, mutatePrivate, wrongModel bool
	cache                           map[string]run.EvaluationResult
}

func (e *fakeEvaluator) Evaluate(_ context.Context, a run.EvaluationAssignment) (run.EvaluationResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.calls++
	if e.fail {
		return run.EvaluationResult{}, errInjected
	}
	if r, ok := e.cache[a.ID]; ok {
		return r, nil
	}
	s, err := e.repo.Load(context.Background(), a.Binding.RunID)
	if err != nil {
		return run.EvaluationResult{}, err
	}
	private := ""
	for i, r := range s.Commit.Registrations {
		if r.Slot.Member.WorkerID == a.WorkerID {
			private = s.PrivateHashes[i]
		}
	}
	h, err := run.Digest(a)
	if err != nil {
		return run.EvaluationResult{}, err
	}
	r := run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: a, AssignmentHash: h, Loss: e.loss, EffectiveTokens: 16, StartedAt: e.clock.Unix(), FinishedAt: e.clock.Unix(), PrivateBeforeHash: private, PrivateAfterHash: private}
	if e.mutatePrivate {
		r.PrivateAfterHash = "different"
	}
	if e.wrongModel {
		r.Assignment.Model.Ref.SHA256 = "different"
	}
	e.cache[a.ID] = r
	return r, nil
}

type harness struct {
	m         Model
	repo      *memoryRepo
	artifacts *memoryArtifacts
	clock     *fakeClock
	alloc     *fakeAllocations
	eval      *fakeEvaluator
	plan      run.Plan
	commit    run.Commit
	policy    EvaluationPolicy
}

func setup(t *testing.T) *harness {
	t.Helper()
	h := &harness{repo: &memoryRepo{receipts: map[string]learner.Receipt{}, rounds: map[uint64]run.Round{}, results: map[uint64]run.EvaluationResult{}, decisions: map[uint64]run.EvaluationDecision{}}, artifacts: &memoryArtifacts{blobs: map[string][]byte{}}, clock: &fakeClock{}, alloc: &fakeAllocations{}}
	h.clock.now.Store(1200)
	h.eval = &fakeEvaluator{repo: h.repo, clock: h.clock, loss: 2, cache: map[string]run.EvaluationResult{}}
	h.m = Model{Repository: h.repo, Artifacts: h.artifacts, Allocations: h.alloc, Clock: h.clock, Evaluator: h.eval}
	for name, dst := range map[string]any{"plan": &h.plan, "commit": &h.commit} {
		b, e := os.ReadFile("../diloco/testdata/" + name + ".json")
		if e != nil {
			t.Fatal(e)
		}
		if e = json.Unmarshal(b, dst); e != nil {
			t.Fatal(e)
		}
	}
	base, e := h.artifacts.Put(context.Background(), "model_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{4}}})
	if e != nil {
		t.Fatal(e)
	}
	h.plan.BaseModel = base
	h.plan.Outer.InnerSteps = 2
	h.plan.Outer.TotalSteps = 6
	h.plan.Outer.LR = 1
	h.plan.Outer.Momentum = .5
	h.plan.SHA256, e = h.plan.Digest()
	if e != nil {
		t.Fatal(e)
	}
	h.commit.Binding.PlanHash = h.plan.SHA256
	for i := range h.commit.Registrations {
		h.commit.Registrations[i].PlanHash = h.plan.SHA256
		h.commit.Registrations[i].BaseModelHash = base.Ref.SHA256
	}
	h.commit.Binding.CommitHash, e = h.commit.Digest()
	if e != nil {
		t.Fatal(e)
	}
	if e = h.commit.ValidateFor(h.plan); e != nil {
		t.Fatal(e)
	}
	h.policy = EvaluationPolicy{WorkerID: "w0", ValidationHash: h.plan.Contract.DatasetHash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 2, TimeoutSeconds: 100}
	return h
}
func (h *harness) state(t *testing.T) State {
	t.Helper()
	s, e := h.repo.Load(context.Background(), h.plan.Cohort.RunID)
	if e != nil {
		t.Fatal(e)
	}
	return s
}
func (h *harness) start(t *testing.T) []run.Assignment {
	t.Helper()
	ctx := context.Background()
	if e := h.m.Stage(ctx, h.plan, h.policy); e != nil {
		t.Fatal(e)
	}
	if e := h.m.Commit(ctx, h.plan.Cohort.RunID, h.commit); e != nil {
		t.Fatal(e)
	}
	a, e := h.m.OpenRound(ctx, h.plan.Cohort.RunID, 100)
	if e != nil {
		t.Fatal(e)
	}
	return a
}
func (h *harness) update(t *testing.T, a run.Assignment, index int) learner.Update {
	t.Helper()
	base, e := h.artifacts.Load(context.Background(), a.BaseModel)
	if e != nil {
		t.Fatal(e)
	}
	local := copyValue(base)
	for i := range local {
		for j, v := range local[i].Values {
			x := float64(v)
			for step := 0; step < 2; step++ {
				x -= .25 * (x - float64(index*2))
			}
			local[i].Values[j] = float32(x)
		}
	}
	delta, e := optimizer.Delta(base, local)
	if e != nil {
		t.Fatal(e)
	}
	ref, e := h.artifacts.Put(context.Background(), "delta_f32", delta)
	if e != nil {
		t.Fatal(e)
	}
	ah, e := run.Digest(a)
	if e != nil {
		t.Fatal(e)
	}
	var norm, max float64
	for _, v := range delta[0].Values {
		norm += float64(v) * float64(v)
		max = math.Max(max, math.Abs(float64(v)))
	}
	r := h.commit.Registrations[index]
	return learner.Update{Format: "mixlab_outer_update_v1", ID: fmt.Sprintf("update-%d-%d", a.Round, index), AssignmentHash: ah, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: r.Slot.Member.ParticipantID, IslandID: r.Slot.Member.IslandID, Contract: a.Contract, Delta: ref, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: h.plan.Contract.ConfigHash, Attempted: 2, Committed: 2, EffectiveTokens: 8, EffectiveExamples: 2, Loss: 2, DeltaL2: math.Sqrt(norm), DeltaMaxAbs: max, StartedAt: h.clock.Unix(), FinishedAt: h.clock.Unix()}
}
func (h *harness) acceptAll(t *testing.T, a []run.Assignment) []learner.Update {
	t.Helper()
	u := make([]learner.Update, len(a))
	for i := len(a) - 1; i >= 0; i-- {
		u[i] = h.update(t, a[i], i)
		if _, e := h.m.Accept(context.Background(), h.plan.Cohort.RunID, u[i]); e != nil {
			t.Fatal(e)
		}
	}
	return u
}
func (h *harness) advance(t *testing.T) {
	t.Helper()
	if e := h.m.Advance(context.Background(), h.plan.Cohort.RunID); e != nil {
		t.Fatal(e)
	}
}
