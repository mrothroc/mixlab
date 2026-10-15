// Package nodeleases adapts node-owned allocations to coordinator liveness.
// It has no recruitment or replacement path, and never assumes a successful
// release request means the physical accelerator has finished cleanup.
package nodeleases

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

type Nodes interface {
	Status(context.Context, run.Allocation) (nodeagent.Lease, error)
	Renew(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
	Release(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error)
}
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
	for _, slot := range s.Plan.Slots {
		l, e := k.Nodes.Status(ctx, slot.Allocation)
		if e != nil {
			return false, e
		}
		if e = match(s, slot, l); e != nil {
			return false, e
		}
		if !terminal && (k.Clock.Unix() >= l.Expires || l.State != nodeagent.Running) {
			return false, k.Fail(ctx, k.RunID, "original learner allocation expired or stopped; replacement forbidden")
		}
		if terminal && l.State == nodeagent.Released {
			continue
		}
		if terminal {
			released = false
		}
		action := "renew"
		ttl := k.TTLSeconds
		if terminal {
			action = "release"
			ttl = 0
		} else if k.Clock.Unix() < l.RenewBy {
			continue
		}
		h, _ := run.Digest(struct {
			Commit, Lease, Action string
			Version               uint64
		}{s.Commit.Binding.CommitHash, l.ID, action, l.Version})
		q := nodeagent.LeaseCommand{IdempotencyKey: h[:32], Lease: l.ID, ExpectedVersion: l.Version, TTLSeconds: ttl}
		if terminal {
			_, e = k.Nodes.Release(ctx, slot.Allocation, q)
		} else {
			_, e = k.Nodes.Renew(ctx, slot.Allocation, q)
		}
		if e != nil {
			return false, e
		}
	}
	return released, nil
}
