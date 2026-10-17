package nodeleases

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

type fake struct {
	mu                   sync.Mutex
	s                    coordinator.State
	leases               map[string]nodeagent.Lease
	now                  int64
	renew, release, fail int
	lost                 bool
}

func (f *fake) Unix() int64                                             { return f.now }
func (f *fake) Load(context.Context, string) (coordinator.State, error) { return f.s, nil }
func (f *fake) Save(context.Context, string, uint64, coordinator.Change) error {
	return errors.New("unexpected save")
}
func (f *fake) Receipt(context.Context, string, string) (learner.Receipt, error) {
	return learner.Receipt{}, errors.New("unexpected receipt")
}
func (f *fake) Status(_ context.Context, a run.Allocation) (nodeagent.Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.leases[a.LeaseID], nil
}
func (f *fake) Renew(_ context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	l := f.leases[a.LeaseID]
	if l.Version != q.ExpectedVersion {
		return l, errors.New("stale lease")
	}
	l.Version++
	l.Expires = f.now + int64(q.TTLSeconds)
	l.RenewBy = f.now + int64(q.TTLSeconds)/2
	f.leases[a.LeaseID] = l
	f.renew++
	if f.lost {
		f.lost = false
		return l, errors.New("lost response")
	}
	return l, nil
}
func (f *fake) Release(_ context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	l := f.leases[a.LeaseID]
	l.State = nodeagent.Releasing
	l.Version++
	f.leases[a.LeaseID] = l
	f.release++
	return l, nil
}
func setup(t *testing.T) (Keeper, *fake) {
	t.Helper()
	f := &fake{now: 1200, leases: map[string]nodeagent.Lease{}}
	var p run.Plan
	var c run.Commit
	for name, dst := range map[string]any{"plan": &p, "commit": &c} {
		b, err := os.ReadFile("../../diloco/testdata/" + name + ".json")
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(b, dst); err != nil {
			t.Fatal(err)
		}
	}
	momentum := p.BaseModel
	momentum.Kind = "outer_state_f32"
	f.s = coordinator.State{Revision: 1, Phase: coordinator.Ready, Plan: p, Commit: &c, Base: p.BaseModel, Momentum: momentum, PrivateHashes: make([]string, len(p.Slots)), Policy: coordinator.EvaluationPolicy{WorkerID: p.Slots[0].Member.WorkerID, ValidationHash: p.Contract.DatasetHash, Split: "validation", BatchTokens: 4, SequenceLength: 4, Batches: 1, TimeoutSeconds: 60}}
	for _, slot := range p.Slots {
		a := slot.Allocation
		f.leases[a.LeaseID] = nodeagent.Lease{ID: a.LeaseID, Node: a.NodeID, Job: a.JobID, Run: p.Cohort.RunID, Coordinator: p.Coordinator.PrincipalID, RunCommitHash: c.Binding.CommitHash, JobKind: "learner", Version: 3, State: nodeagent.Running, Expires: 1300, RenewBy: 1190}
	}
	k := Keeper{Repository: f, Nodes: f, Clock: f, RunID: p.Cohort.RunID, TTLSeconds: 300, Fail: func(context.Context, string, string) error {
		f.fail++
		f.s.Phase = coordinator.Failed
		f.s.Failure = "allocation expired"
		return nil
	}}
	return k, f
}
func TestKeeperRenewLostResponseAndTerminalCleanup(t *testing.T) {
	k, f := setup(t)
	ctx := context.Background()
	f.lost = true
	if _, err := k.Tick(ctx); err == nil {
		t.Fatal("expected lost response")
	}
	if _, err := k.Tick(ctx); err != nil {
		t.Fatal(err)
	}
	if f.renew != len(f.leases) {
		t.Fatal("duplicate renewal after reread")
	}
	for _, slot := range f.s.Plan.Slots {
		if err := k.Ready(ctx, slot); err != nil {
			t.Fatal(err)
		}
	}
	f.s.Phase = coordinator.Failed
	f.s.Failure = "test"
	if released, err := k.Tick(ctx); err != nil || released {
		t.Fatal("release request confused with cleanup", released, err)
	}
	for id, l := range f.leases {
		l.State = nodeagent.Released
		f.leases[id] = l
	}
	if released, err := k.Tick(ctx); err != nil || !released {
		t.Fatal(released, err)
	}
}
func TestKeeperExpiryFailsWithoutReplacement(t *testing.T) {
	k, f := setup(t)
	f.now = 1300
	if _, err := k.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.fail != 1 || f.renew != 0 || f.s.Phase != coordinator.Failed {
		t.Fatal("expired allocation was renewed or ignored")
	}
}
func TestKeeperRequiresExactAuthority(t *testing.T) {
	k, f := setup(t)
	slot := f.s.Plan.Slots[0]
	l := f.leases[slot.Allocation.LeaseID]
	l.RunCommitHash = "wrong"
	f.leases[l.ID] = l
	if k.Ready(context.Background(), slot) == nil {
		t.Fatal("foreign handoff accepted")
	}
}
