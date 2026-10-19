package learner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
)

var errCrash = errors.New("injected crash")

type fakeEngine struct {
	v       Snapshot
	calls   *int
	fail    *bool
	skipped bool
}

func clone[T any](v T) T                                 { b, _ := json.Marshal(v); var out T; _ = json.Unmarshal(b, &out); return out }
func (e *fakeEngine) Snapshot() (Snapshot, error)        { return clone(e.v), nil }
func (e *fakeEngine) Restore(v Snapshot) error           { e.v = clone(v); return nil }
func (e *fakeEngine) Install(v []optimizer.Tensor) error { e.v.Model = clone(v); return nil }
func (e *fakeEngine) Close()                             {}
func (e *fakeEngine) Train(_ context.Context, n uint64) (Counts, error) {
	*e.calls++
	for i := uint64(0); i < n; i++ {
		e.v.Replay.AttemptStep++
		// Model and synthetic private momentum depend on the restored cursor.
		e.v.Model[0].Values[0] -= float32(e.v.Replay.AttemptStep) * .125
		e.v.Optimizer = []byte(fmt.Sprintf("moment-%d", e.v.Replay.AttemptStep))
		e.v.Sampler = []byte(fmt.Sprintf("cursor-%d", e.v.Replay.AttemptStep))
		if *e.fail {
			*e.fail = false
			return Counts{}, errCrash
		}
	}
	c := Counts{Attempted: n, Committed: n, Tokens: 8 * n, Examples: 2 * n, Loss: 2}
	if e.skipped {
		c.Skipped = 1
		c.Committed--
	}
	return c, nil
}

type fakePeer struct {
	base                  []byte
	starts                []private.Start
	updates               []private.Update
	receipt               private.Receipt
	failStart, failSubmit bool
	deltas                [][]byte
}

func (p *fakePeer) Read(_ context.Context, _ run.Assignment, _ run.Blob, w io.Writer) error {
	_, err := w.Write(p.base)
	return err
}
func (p *fakePeer) Write(_ context.Context, _ run.Assignment, _ run.Blob, r io.Reader) error {
	b, err := io.ReadAll(r)
	p.deltas = append(p.deltas, b)
	return err
}
func (p *fakePeer) Start(_ context.Context, s private.Start) error {
	p.starts = append(p.starts, s)
	if p.failStart {
		p.failStart = false
		return errCrash
	}
	return nil
}
func (p *fakePeer) Submit(_ context.Context, u private.Update) (private.Receipt, error) {
	p.updates = append(p.updates, u)
	h, _ := run.Digest(u)
	p.receipt = private.Receipt{Format: "mixlab_update_receipt_v1", Binding: u.Binding, UpdateID: u.ID, UpdateHash: h, PrivateHash: u.PrivateHash}
	if p.failSubmit {
		p.failSubmit = false
		return private.Receipt{}, errCrash
	}
	return p.receipt, nil
}

type fixture struct {
	r     Runtime
	a     run.Assignment
	p     *fakePeer
	calls int
	fail  bool
	skip  bool
}

func setup(t *testing.T) *fixture {
	t.Helper()
	return setupAt(t, t.TempDir())
}

func setupAt(t *testing.T, root string) *fixture {
	t.Helper()
	f := &fixture{p: &fakePeer{}}
	b, err := os.ReadFile("../diloco/testdata/assignment.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &f.a); err != nil {
		t.Fatal(err)
	}
	f.a.InnerSteps = 2
	model := []optimizer.Tensor{{Name: "w", Shape: []int{1}, Values: []float32{4}}}
	f.p.base, err = encodeModel(model)
	if err != nil {
		t.Fatal(err)
	}
	f.a.BaseModel.Ref = ref(f.p.base)
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "private")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	if err = path.Ensure(); err != nil {
		t.Fatal(err)
	}
	s, err := Open(path, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	f.r = Runtime{Store: s, Peer: f.p, Now: func() int64 { return 1300 }, Scope: Scope{RunID: f.a.Binding.RunID, PlanHash: f.a.Binding.PlanHash, WorkerID: f.a.WorkerID, PartitionID: f.a.PartitionID, ParticipantID: "node0", IslandID: "island0", Contract: f.a.Contract, Coordinator: f.a.Coordinator}}
	f.r.Scope.CohortHash = f.a.Binding.CohortHash
	f.r.Factory = func(model []optimizer.Tensor) (Engine, error) {
		return &fakeEngine{calls: &f.calls, fail: &f.fail, skipped: f.skip, v: Snapshot{Model: clone(model), Optimizer: []byte("moment-0"), Sampler: []byte("cursor-0"), Schedule: []byte("schedule"), Metadata: []byte("metadata"), Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: f.a.Binding.RunID, WorkerID: f.a.WorkerID, PartitionID: f.a.PartitionID, Seed: 7}, Precision: precision.Policy{ComputeDType: "float32", Backend: "metal"}}}, nil
	}
	f.a.PrecisionHash, _ = (precision.Policy{ComputeDType: "float32", Backend: "metal"}).Digest()
	return f
}
func (f *fixture) run(t *testing.T) private.Receipt {
	t.Helper()
	r, err := f.r.Execute(context.Background(), f.a)
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func TestRuntimeReplayCompletionAndLostReceipt(t *testing.T) {
	for _, point := range []string{"start", "mid-work", "completed", "submit", "receipt"} {
		t.Run(point, func(t *testing.T) {
			f := setup(t)
			switch point {
			case "start":
				f.p.failStart = true
			case "mid-work":
				f.fail = true
			case "submit":
				f.p.failSubmit = true
			default:
				f.r.Store.fault = func(p string) error {
					if p == point {
						return errCrash
					}
					return nil
				}
			}
			if _, err := f.r.Execute(context.Background(), f.a); !errors.Is(err, errCrash) {
				t.Fatal("expected crash", err)
			}
			f.r.Store.fault = nil
			_, j, err := f.r.Store.load()
			if err != nil {
				t.Fatal(err)
			}
			f.r.Scope.ExpectedPrivateHash, _ = run.Digest(j.Start)
			r := f.run(t)
			want := 1
			if point == "mid-work" {
				want = 2
			}
			if f.calls != want {
				t.Fatalf("train calls %d want %d", f.calls, want)
			}
			if again := f.run(t); again != r || f.calls != want {
				t.Fatal("repeat trained twice")
			}
			if len(f.p.updates) > 1 && !reflect.DeepEqual(f.p.updates[0], f.p.updates[1]) {
				t.Fatal("lost receipt changed update")
			}
			for _, s := range f.p.starts {
				if s != f.p.starts[0] {
					t.Fatal("start replay changed")
				}
			}
			_, j, err = f.r.Store.load()
			if err != nil {
				t.Fatal(err)
			}
			v, err := f.r.Store.restore(context.Background(), *j.Completed)
			if err != nil {
				t.Fatal(err)
			}
			if j.Completed.Replay.AttemptStep != 2 || string(v.Sampler) != "cursor-2" {
				t.Fatal("cursor/RNG not restored")
			}
		})
	}
}
func TestRuntimeSuccessorKeepsPrivateMoments(t *testing.T) {
	f := setup(t)
	r := f.run(t)
	f.a.Round = 1
	f.a.BaseRound = 1
	f.a.ID = "round1"
	f.a.PreviousPrivateHash = r.PrivateHash
	f.p.base, _ = encodeModel([]optimizer.Tensor{{Name: "w", Shape: []int{1}, Values: []float32{10}}})
	f.a.BaseModel.Ref = ref(f.p.base)
	f.run(t)
	_, j, err := f.r.Store.load()
	if err != nil {
		t.Fatal(err)
	}
	v, err := f.r.Store.restore(context.Background(), *j.Completed)
	if err != nil {
		t.Fatal(err)
	}
	if v.Model[0].Values[0] != 9.125 || v.Replay.AttemptStep != 4 || string(v.Optimizer) != "moment-4" {
		t.Fatalf("state lost: %+v", v)
	}
}
func TestRuntimeFailsClosed(t *testing.T) {
	for _, mode := range []string{"missing", "scope", "skip", "changed", "corrupt", "budget"} {
		t.Run(mode, func(t *testing.T) {
			f := setup(t)
			switch mode {
			case "missing":
				f.r.Scope.ExpectedPrivateHash = f.a.Contract.ConfigHash
			case "scope":
				f.a.PartitionID = "other"
			case "skip":
				f.skip = true
			case "changed":
				f.run(t)
				f.a.ExpiresAt++
			case "corrupt":
				f.run(t)
				_, j, _ := f.r.Store.load()
				if err := os.WriteFile(filepath.Join(f.r.Store.path.Dir(), j.Completed.Sampler.SHA256), []byte("bad"), 0600); err != nil {
					t.Fatal(err)
				}
			case "budget":
				f.r.Store.limit = 128
			}
			if _, err := f.r.Execute(context.Background(), f.a); err == nil {
				t.Fatal("invalid recovery succeeded")
			}
		})
	}
}
