package nodeagent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

// CheckApprovedWorker prevents startup on an interrupted approval transaction.
func (s *Store) CheckApprovedWorker(build string) error {
	_, r, err := s.load()
	if err != nil {
		return err
	}
	_, p, err := s.loadProfile()
	if err != nil {
		return err
	}
	if p.Generation != r.CapabilityGeneration || p.Probe.BuildID != build {
		return fmt.Errorf("worker approval publication incomplete; retry agent reapprove while stopped")
	}
	return nil
}

// ReapproveWorker runs only during offline local administration. Profile-first
// publication leaves capabilities fenced until an interrupted approval is retried.
// The callback publishes the owning application's executable pins, never a lease.
func (s *Store) ReapproveWorker(ctx context.Context, probe func(context.Context) (workerprobe.Report, error), publish func() error, now time.Time) error {
	if probe == nil || publish == nil {
		return fmt.Errorf("probe and installation publisher required")
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		leases, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Active != "" {
			return fmt.Errorf("cannot reapprove a node with an active or uncleared lease")
		}
		old, p, err := s.loadProfile()
		if err != nil {
			return err
		}
		if p.Generation != r.CapabilityGeneration && p.Generation != r.CapabilityGeneration+1 {
			return fmt.Errorf("profile generation diverged")
		}
		report, err := probe(ctx)
		if err != nil {
			return err
		}
		p.Probe, p.ProbeObservedAt = report, now.Unix()
		p.Generation = r.CapabilityGeneration + 1
		if err := p.Validate(); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		b, err := json.Marshal(p)
		if err != nil {
			return err
		}
		if err := s.path.CompareAndSwap(profileFile, old, b); err != nil {
			return err
		}
		if err := publish(); err != nil {
			return err
		}
		r.CapabilityGeneration = p.Generation
		r.NodeVersion++
		return s.save(leases, r)
	})
}
