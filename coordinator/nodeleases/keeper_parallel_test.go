package nodeleases

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

type controlledNodes struct {
	Nodes
	status  func(context.Context, run.Allocation) (nodeagent.Lease, error)
	renew   func(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
	release func(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
}

func (n controlledNodes) Status(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
	if n.status != nil {
		return n.status(ctx, a)
	}
	return n.Nodes.Status(ctx, a)
}

func (n controlledNodes) Renew(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	if n.renew != nil {
		return n.renew(ctx, a, q)
	}
	return n.Nodes.Renew(ctx, a, q)
}

func (n controlledNodes) Release(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	if n.release != nil {
		return n.release(ctx, a, q)
	}
	return n.Nodes.Release(ctx, a, q)
}

func TestKeeperSlowFirstPeerDoesNotStarveMaintenance(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		name := "renew"
		if terminal {
			name = "release"
		}
		t.Run(name, func(t *testing.T) {
			k, f := setup(t)
			if terminal {
				f.s.Phase, f.s.Failure = coordinator.Failed, "test"
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			first := f.s.Plan.Slots[0].Allocation
			entered, unblock := make(chan struct{}), make(chan struct{})
			maintained := make(chan struct{}, len(f.s.Plan.Slots))
			failure := errors.New("first node unavailable")
			k.Nodes = controlledNodes{Nodes: f,
				status: func(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
					if a == first {
						close(entered)
						select {
						case <-unblock:
							return nodeagent.Lease{}, failure
						case <-ctx.Done():
							return nodeagent.Lease{}, ctx.Err()
						}
					}
					return f.Status(ctx, a)
				},
				renew: func(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
					l, err := f.Renew(ctx, a, q)
					maintained <- struct{}{}
					return l, err
				},
				release: func(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
					l, err := f.Release(ctx, a, q)
					maintained <- struct{}{}
					return l, err
				},
			}
			done := make(chan error, 1)
			go func() {
				released, err := k.Tick(ctx)
				if released {
					err = errors.New("failed peer reported physically released")
				}
				done <- err
			}()
			<-entered
			for range len(f.s.Plan.Slots) - 1 {
				select {
				case <-maintained:
				case <-ctx.Done():
					<-done
					t.Fatal("slow first node starved another allocation")
				}
			}
			select {
			case err := <-done:
				t.Fatalf("returned without joining slow node: %v", err)
			default:
			}
			close(unblock)
			if err := <-done; !errors.Is(err, failure) {
				t.Fatalf("lost peer failure: %v", err)
			}
		})
	}
}

func TestKeeperParallelBoundAndCancellationJoin(t *testing.T) {
	k, f := setup(t)
	s := f.s
	// Exercise the scheduling helper beyond one batch without changing plan validation.
	s.Plan.Slots = make([]run.Slot, 19)
	for i := range s.Plan.Slots {
		s.Plan.Slots[i] = f.s.Plan.Slots[0]
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan struct{}, len(s.Plan.Slots))
	var active, peak, calls atomic.Int32
	k.Nodes = controlledNodes{Nodes: f, status: func(ctx context.Context, _ run.Allocation) (nodeagent.Lease, error) {
		calls.Add(1)
		n := active.Add(1)
		defer active.Add(-1)
		for old := peak.Load(); n > old; old = peak.Load() {
			if peak.CompareAndSwap(old, n) {
				break
			}
		}
		entered <- struct{}{}
		<-ctx.Done()
		return nodeagent.Lease{}, ctx.Err()
	}}
	done := make(chan []leaseResult, 1)
	go func() { done <- k.maintainSlots(ctx, s, false) }()
	for range 8 {
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			cancel()
			<-done
			t.Fatal("bounded workers did not start")
		}
	}
	cancel()
	results := <-done
	if active.Load() != 0 || peak.Load() != 8 || calls.Load() != 8 {
		t.Fatalf("workers not bounded/joined: active=%d peak=%d calls=%d", active.Load(), peak.Load(), calls.Load())
	}
	for i, result := range results {
		if !errors.Is(result.err, context.Canceled) {
			t.Fatalf("slot %d lost cancellation: %v", i, result.err)
		}
	}
}

func TestKeeperJoinsErrorsInSlotOrder(t *testing.T) {
	k, f := setup(t)
	failures := make(map[run.Allocation]error)
	for i, slot := range f.s.Plan.Slots {
		failures[slot.Allocation] = errors.New(strings.Repeat("x", i+1) + " node failure")
	}
	k.Nodes = controlledNodes{Nodes: f, status: func(_ context.Context, a run.Allocation) (nodeagent.Lease, error) {
		return nodeagent.Lease{}, failures[a]
	}}
	_, err := k.Tick(context.Background())
	var ordered []error
	for _, slot := range f.s.Plan.Slots {
		failure := failures[slot.Allocation]
		if !errors.Is(err, failure) {
			t.Fatalf("missing peer failure: %v", err)
		}
		ordered = append(ordered, failure)
	}
	if err.Error() != errors.Join(ordered...).Error() {
		t.Fatalf("nondeterministic error order: %v", err)
	}
}

func TestKeeperServicesEverySlotBeyondFirstBatch(t *testing.T) {
	k, f := setup(t)
	s := f.s
	s.Plan.Slots = make([]run.Slot, 19)
	for i := range s.Plan.Slots {
		s.Plan.Slots[i] = f.s.Plan.Slots[0]
	}
	var calls atomic.Int32
	k.Nodes = controlledNodes{Nodes: f, status: func(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
		calls.Add(1)
		l, err := f.Status(ctx, a)
		l.RenewBy = f.now + 1
		return l, err
	}}
	for i, result := range k.maintainSlots(context.Background(), s, false) {
		if result.err != nil || result.lost {
			t.Fatalf("slot %d failed: %+v", i, result)
		}
	}
	if calls.Load() != int32(len(s.Plan.Slots)) {
		t.Fatalf("skipped queued slots: calls=%d", calls.Load())
	}
}

func TestKeeperForeignPeerDoesNotPreventHealthyRenewal(t *testing.T) {
	k, f := setup(t)
	first := f.s.Plan.Slots[0].Allocation
	l := f.leases[first.LeaseID]
	l.RunCommitHash = "wrong"
	f.leases[first.LeaseID] = l
	if _, err := k.Tick(context.Background()); err == nil {
		t.Fatal("foreign handoff accepted")
	}
	if f.renew != len(f.s.Plan.Slots)-1 || f.leases[first.LeaseID].Version != l.Version {
		t.Fatal("invalid peer was renewed or healthy peer was skipped")
	}
}
