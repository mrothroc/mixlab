// Package nodeleases adapts node-owned allocations to coordinator liveness.
// It has no recruitment or replacement path, and never assumes a successful
// release request means the physical accelerator has finished cleanup.
package nodeleases

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

// Nodes must support concurrent calls for different allocations and honor ctx.
type Nodes interface {
	Status(context.Context, run.Allocation) (nodeagent.Lease, error)
	Renew(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
	Release(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
}

// ErrReleaseConflict permits a fresh status read, not an assumption of cleanup.
// Transport adapters must not classify authentication or availability errors here.
var ErrReleaseConflict = errors.New("allocation release conflict")

type Keeper struct {
	Repository coordinator.Repository
	Nodes      Nodes
	Clock      coordinator.Clock
	RunID      string
	TTLSeconds int
	Fail       func(context.Context, string, string) error
}

func (k Keeper) state(ctx context.Context) (coordinator.State, error) {
	if k.Repository == nil || k.Nodes == nil || k.Clock == nil || k.Fail == nil || k.TTLSeconds < 10 || k.TTLSeconds > 3600 {
		return coordinator.State{}, fmt.Errorf("allocation keeper ports and bounded TTL required")
	}
	s, err := k.Repository.Load(ctx, k.RunID)
	if err != nil {
		return s, err
	}
	if err = s.Validate(); err != nil {
		return s, err
	}
	if s.Plan.Cohort.RunID != k.RunID || s.Commit == nil {
		return s, fmt.Errorf("committed keeper run required")
	}
	return s, nil
}
func match(s coordinator.State, slot run.Slot, l nodeagent.Lease) error {
	if l.ID != slot.Allocation.LeaseID || l.Node != slot.Allocation.NodeID || l.Job != slot.Allocation.JobID || l.Run != s.Plan.Cohort.RunID || l.Coordinator != s.Plan.Coordinator.PrincipalID || l.RunCommitHash != s.Commit.Binding.CommitHash || l.JobKind != "learner" || l.Version == 0 {
		return fmt.Errorf("allocation authority handoff missing or changed")
	}
	return nil
}
func (k Keeper) Ready(ctx context.Context, slot run.Slot) error {
	s, err := k.state(ctx)
	if err != nil {
		return err
	}
	found := false
	for _, v := range s.Plan.Slots {
		found = found || v == slot
	}
	if !found || s.Phase == coordinator.Failed || s.Phase == coordinator.Complete {
		return fmt.Errorf("inactive or foreign allocation")
	}
	l, err := k.Nodes.Status(ctx, slot.Allocation)
	if err != nil {
		return err
	}
	if err = match(s, slot, l); err != nil {
		return err
	}
	if l.State != nodeagent.Running || k.Clock.Unix() >= l.Expires {
		return fmt.Errorf("original allocation expired or stopped")
	}
	return nil
}

// Tick runs independently of the submitter. Lost renewal responses are resolved
// by rereading the lease; the key is deterministic for an exact version command.
func (k Keeper) Tick(ctx context.Context) (released bool, err error) {
	s, err := k.state(ctx)
	if err != nil {
		return false, err
	}
	terminal := s.Phase == coordinator.Failed || s.Phase == coordinator.Complete
	released = terminal
	lost := false
	for _, result := range k.maintainSlots(ctx, s, terminal) {
		released = released && result.released
		lost = lost || result.lost
		err = errors.Join(err, result.err)
	}
	// Serialize the run transition after joining all allocation effects.
	if lost {
		err = errors.Join(err, k.Fail(ctx, k.RunID, "original learner allocation expired or stopped; replacement forbidden"))
	}
	return released && err == nil, err
}

type leaseResult struct {
	released bool
	lost     bool
	err      error
}

func (k Keeper) maintainSlots(ctx context.Context, s coordinator.State, terminal bool) []leaseResult {
	results := make([]leaseResult, len(s.Plan.Slots))
	workers := min(8, len(results))
	work := make(chan int, len(results))
	for i := range results {
		work <- i
	}
	close(work)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				if err := ctx.Err(); err != nil {
					results[i].err = err
					continue
				}
				results[i] = k.maintainSlot(ctx, s, s.Plan.Slots[i], terminal)
			}
		}()
	}
	wg.Wait()
	return results
}

func (k Keeper) maintainSlot(ctx context.Context, s coordinator.State, slot run.Slot, terminal bool) leaseResult {
	l, err := k.Nodes.Status(ctx, slot.Allocation)
	if err != nil {
		return leaseResult{err: err}
	}
	if err = match(s, slot, l); err != nil {
		return leaseResult{err: err}
	}
	if !terminal && (k.Clock.Unix() >= l.Expires || l.State != nodeagent.Running) {
		return leaseResult{lost: true}
	}
	if terminal && l.State == nodeagent.Released {
		return leaseResult{released: true}
	}
	action, ttl := "renew", k.TTLSeconds
	if terminal {
		action, ttl = "release", 0
	} else if k.Clock.Unix() < l.RenewBy {
		return leaseResult{}
	}
	h, _ := run.Digest(struct {
		Commit, Lease, Action string
		Version               uint64
	}{s.Commit.Binding.CommitHash, l.ID, action, l.Version})
	q := nodeagent.LeaseCommand{IdempotencyKey: h[:32], Lease: l.ID, ExpectedVersion: l.Version, TTLSeconds: ttl}
	if s.Phase == coordinator.Complete {
		attempt, _, e := s.CurrentAttempt(slot.Member.WorkerID)
		if e != nil {
			return leaseResult{err: e}
		}
		q.CompletedAttempt = attempt
	}
	if terminal {
		_, err = k.Nodes.Release(ctx, slot.Allocation, q)
		if errors.Is(err, ErrReleaseConflict) {
			return k.reconcileRelease(ctx, s, slot, q, err)
		}
	} else {
		_, err = k.Nodes.Renew(ctx, slot.Allocation, q)
	}
	return leaseResult{err: err}
}

func (k Keeper) reconcileRelease(ctx context.Context, s coordinator.State, slot run.Slot, q nodeagent.LeaseCommand, conflict error) leaseResult {
	// Cleanup can finish between Status and Release. Only a new authenticated
	// observation of that exact allocation proves the conflicting release is moot.
	l, err := k.Nodes.Status(ctx, slot.Allocation)
	if err == nil {
		err = match(s, slot, l)
	}
	if err != nil {
		return leaseResult{err: errors.Join(conflict, err)}
	}
	if l.State != nodeagent.Released || l.Version <= q.ExpectedVersion {
		return leaseResult{err: conflict}
	}
	if q.CompletedAttempt != "" && (l.CompletedAttempt != q.CompletedAttempt || l.CleanupReason != "completed") {
		return leaseResult{err: conflict}
	}
	return leaseResult{released: true}
}
