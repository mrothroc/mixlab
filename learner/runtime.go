package learner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Scope is installed by the hosting assignment, never inferred from a peer's
// round response. ExpectedPrivateHash is required on a replacement attempt.
type Scope struct {
	RunID, PlanHash, WorkerID, PartitionID, ParticipantID, IslandID string
	CohortHash                                                      string
	Generation                                                      uint64
	Contract                                                        run.Contract
	Coordinator                                                     run.Coordinator
	ExpectedPrivateHash                                             string
}

type Runtime struct {
	mu      sync.Mutex
	Store   *Store
	Peer    Peer
	Factory Factory
	Scope   Scope
	Now     func() int64
}

// Execute completes or replays one immutable assignment. No optimizer state or
// sampler bytes leave the worker. All retries reuse the published update bytes.
func (r *Runtime) Execute(ctx context.Context, a run.Assignment) (receipt private.Receipt, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Store == nil || r.Peer == nil || r.Factory == nil {
		return receipt, fmt.Errorf("learner ports required")
	}
	if err = a.Validate(); err != nil {
		return
	}
	s := r.Scope
	if a.Binding.RunID != s.RunID || a.Binding.PlanHash != s.PlanHash || a.Binding.CohortHash != s.CohortHash || a.Binding.Generation != s.Generation || a.WorkerID != s.WorkerID || a.PartitionID != s.PartitionID || a.Contract != s.Contract || a.Coordinator != s.Coordinator || s.ParticipantID == "" || s.IslandID == "" {
		return receipt, fmt.Errorf("round differs from local learner authority")
	}
	if r.Now == nil {
		r.Now = func() int64 { return time.Now().Unix() }
	}
	err = r.Store.locked(ctx, func() error { var e error; receipt, e = r.execute(ctx, a); return e })
	if err == nil {
		r.Scope.ExpectedPrivateHash = ""
	}
	return
}

func (r *Runtime) execute(ctx context.Context, a run.Assignment) (private.Receipt, error) {
	old, j, err := r.Store.load()
	if err != nil && !os.IsNotExist(err) {
		return private.Receipt{}, err
	}
	absent := os.IsNotExist(err)
	if absent && (a.Round != 0 || a.PreviousPrivateHash != "" || r.Scope.ExpectedPrivateHash != "") {
		return private.Receipt{}, fmt.Errorf("required private history is missing; weights-only restart is not recovery")
	}
	if absent {
		if err = r.Store.collect(nil); err != nil {
			return private.Receipt{}, err
		}
	}
	if !absent {
		if err = r.checkHistory(j, a); err != nil {
			return private.Receipt{}, err
		}
		if err = r.Store.collect(&j); err != nil {
			return private.Receipt{}, err
		}
		if j.Assignment == a && j.Marker != nil {
			return r.submit(ctx, old, j)
		}
	}
	if r.Now() < a.IssuedAt || r.Now() >= a.ExpiresAt {
		return private.Receipt{}, fmt.Errorf("round outside execution deadline")
	}
	ctx, cancel := context.WithDeadline(ctx, time.Now().Add(time.Duration(a.ExpiresAt-r.Now())*time.Second))
	defer cancel()
	base, err := r.model(ctx, a)
	if err != nil {
		return private.Receipt{}, err
	}
	engine, err := r.Factory(base)
	if err != nil {
		return private.Receipt{}, err
	}
	defer engine.Close()
	if !absent && j.Assignment == a {
		v, e := r.Store.restore(ctx, j.Start)
		if e != nil {
			return private.Receipt{}, e
		}
		if e = engine.Restore(v); e != nil {
			return private.Receipt{}, e
		}
	} else {
		if !absent {
			v, e := r.Store.restore(ctx, *j.Completed)
			if e != nil {
				return private.Receipt{}, e
			}
			if e = engine.Restore(v); e != nil {
				return private.Receipt{}, e
			}
			if e = engine.Install(base); e != nil {
				return private.Receipt{}, e
			}
		}
		v, e := engine.Snapshot()
		if e != nil {
			return private.Receipt{}, e
		}
		h, e := run.Digest(a)
		if e != nil {
			return private.Receipt{}, e
		}
		start, e := r.Store.snapshot(ctx, a, "update-"+h, "start", v, Counts{})
		if e != nil {
			return private.Receipt{}, e
		}
		j = journal{Format: "mixlab_learner_journal_v1", Assignment: a, Start: start}
		if e = r.Store.save(old, j, "start"); e != nil {
			return private.Receipt{}, e
		}
		old, _, e = r.Store.load()
		if e != nil {
			return private.Receipt{}, e
		}
		if e = r.Store.collect(&j); e != nil {
			return private.Receipt{}, e
		}
	}
	ph, err := run.Digest(j.Start)
	if err != nil {
		return private.Receipt{}, err
	}
	if err = r.Peer.Start(ctx, private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: j.Start.AssignmentHash, PrivateHash: ph}); err != nil {
		return private.Receipt{}, err
	}
	started := r.Now()
	c, err := engine.Train(ctx, a.InnerSteps)
	if err != nil {
		return private.Receipt{}, err
	}
	if c.Attempted != a.InnerSteps || c.Committed != a.InnerSteps || c.Skipped != 0 || c.Tokens == 0 || c.Examples == 0 || math.IsNaN(c.Loss) || math.IsInf(c.Loss, 0) || c.Loss < 0 {
		return private.Receipt{}, fmt.Errorf("inner window did not commit exactly H finite optimizer attempts")
	}
	v, err := engine.Snapshot()
	if err != nil {
		return private.Receipt{}, err
	}
	p, err := r.Store.snapshot(ctx, a, j.Start.UpdateID, "completed", v, c)
	if err != nil {
		return private.Receipt{}, err
	}
	u, err := r.update(ctx, a, j.Start.UpdateID, base, v.Model, p, c, started)
	if err != nil {
		return private.Receipt{}, err
	}
	uh, err := run.Digest(u)
	if err != nil {
		return private.Receipt{}, err
	}
	m := private.Completion{Format: "mixlab_learner_completion_v1", AssignmentHash: j.Start.AssignmentHash, UpdateID: u.ID, PrivateHash: u.PrivateHash, UpdateHash: uh}
	m.SHA256, err = m.Digest()
	if err != nil {
		return private.Receipt{}, err
	}
	j.Completed = &p
	j.Update = &u
	j.Marker = &m
	if err = r.Store.save(old, j, "completed"); err != nil {
		return private.Receipt{}, err
	}
	old, _, err = r.Store.load()
	if err != nil {
		return private.Receipt{}, err
	}
	return r.submit(ctx, old, j)
}

func (r *Runtime) checkHistory(j journal, a run.Assignment) error {
	if expected := r.Scope.ExpectedPrivateHash; expected != "" {
		start, err := run.Digest(j.Start)
		if err != nil {
			return err
		}
		// A locally completed marker can follow the acknowledged start even
		// when the submission response was lost. Both remain in this journal.
		if expected != start && (j.Marker == nil || expected != j.Marker.PrivateHash) && expected != j.Assignment.PreviousPrivateHash {
			return fmt.Errorf("private checkpoint differs from recovery authorization")
		}
	}
	if j.Assignment == a {
		return nil
	}
	if j.Assignment.Binding != a.Binding || j.Assignment.WorkerID != a.WorkerID || j.Assignment.PartitionID != a.PartitionID || j.Assignment.Contract != a.Contract || j.Assignment.Coordinator != a.Coordinator || j.Assignment.PrecisionHash != a.PrecisionHash || j.Assignment.InnerSteps != a.InnerSteps || j.Assignment.Round+1 != a.Round || j.Completed == nil || j.Marker == nil || a.PreviousPrivateHash != j.Marker.PrivateHash {
		return fmt.Errorf("assignment is neither exact replay nor committed successor")
	}
	return nil
}

func (r *Runtime) model(ctx context.Context, a run.Assignment) ([]optimizer.Tensor, error) {
	if a.BaseModel.Ref.Bytes > r.Store.limit {
		return nil, fmt.Errorf("base model exceeds private budget")
	}
	var b bytes.Buffer
	if err := r.Peer.Read(ctx, a, a.BaseModel, &boundedWriter{dst: &b, left: a.BaseModel.Ref.Bytes}); err != nil {
		return nil, err
	}
	if err := artifact.Copy(ctx, io.Discard, bytes.NewReader(b.Bytes()), a.BaseModel.Ref); err != nil {
		return nil, err
	}
	return tensors.Read(bytes.NewReader(b.Bytes()), a.BaseModel.Ref.Bytes)
}

type boundedWriter struct {
	dst  io.Writer
	left uint64
}

func (w *boundedWriter) Write(b []byte) (int, error) {
	if uint64(len(b)) > w.left {
		return 0, fmt.Errorf("artifact response exceeds declared size")
	}
	n, err := w.dst.Write(b)
	w.left -= uint64(n)
	return n, err
}

func (r *Runtime) update(ctx context.Context, a run.Assignment, id string, base, local []optimizer.Tensor, p private.PrivateState, c Counts, started int64) (private.Update, error) {
	d, err := optimizer.Delta(base, local)
	if err != nil {
		return private.Update{}, err
	}
	b, err := encodeModel(d)
	if err != nil {
		return private.Update{}, err
	}
	ref, err := r.Store.put(ctx, b)
	if err != nil {
		return private.Update{}, err
	}
	ph, err := run.Digest(p)
	if err != nil {
		return private.Update{}, err
	}
	ah, err := run.Digest(a)
	if err != nil {
		return private.Update{}, err
	}
	var norm, max float64
	for _, t := range d {
		for _, x := range t.Values {
			v := float64(x)
			norm += v * v
			max = math.Max(max, math.Abs(v))
		}
	}
	u := private.Update{Format: "mixlab_outer_update_v1", ID: id, AssignmentHash: ah, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: r.Scope.ParticipantID, IslandID: r.Scope.IslandID, Contract: a.Contract, Delta: run.Blob{Ref: ref, Kind: "delta_f32"}, DType: "float32", TensorCount: uint64(len(d)), PrecisionHash: a.PrecisionHash, PrivateHash: ph, Attempted: c.Attempted, Committed: c.Committed, Skipped: c.Skipped, EffectiveTokens: c.Tokens, EffectiveExamples: c.Examples, Loss: c.Loss, DeltaL2: math.Sqrt(norm), DeltaMaxAbs: max, StartedAt: started, FinishedAt: r.Now()}
	return u, u.ValidateFor(a)
}

func (r *Runtime) submit(ctx context.Context, old []byte, j journal) (private.Receipt, error) {
	if _, err := r.Store.restore(ctx, *j.Completed); err != nil {
		return private.Receipt{}, err
	}
	f, err := r.Store.blobs.OpenVerified(ctx, j.Update.Delta.Ref)
	if err != nil {
		return private.Receipt{}, err
	}
	defer func() { _ = f.Close() }()
	if j.Receipt != nil {
		return *j.Receipt, nil
	}
	if lookup, ok := r.Peer.(ReceiptLookup); ok {
		if receipt, e := lookup.Receipt(ctx, *j.Update); e == nil {
			j.Receipt = &receipt
			if e = r.Store.save(old, j, "receipt"); e != nil {
				return private.Receipt{}, e
			}
			return receipt, nil
		} else if !errors.Is(e, private.ErrReceiptPending) {
			return private.Receipt{}, e
		}
	}
	if err = r.Peer.Write(ctx, j.Assignment, j.Update.Delta, f); err != nil {
		return private.Receipt{}, err
	}
	receipt, err := r.Peer.Submit(ctx, *j.Update)
	if err != nil {
		return receipt, err
	}
	j.Receipt = &receipt
	if err = r.Store.save(old, j, "receipt"); err != nil {
		return private.Receipt{}, err
	}
	return receipt, nil
}
