package nodeagent

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

// PreparationFence authorizes only cancellation of an unstarted job. Hosting
// persists the tombstone before returning a no-child outcome. The caller must
// apply that outcome and finish credential/transport cleanup before release.
func (s *Store) PreparationFence(ctx context.Context, id string) (out contract.Fence, err error) {
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || l.Job != id || r.Approval != nil || (!r.Job.CancelRequested && l.State != Releasing) {
			return fmt.Errorf("unstarted cancellation/expiry intent required")
		}
		out = contract.Fence{JobID: r.Job.ID, AttemptID: r.Job.Attempt, ManifestHash: r.Job.ManifestHash}
		return nil
	})
	return out, err
}

// CancellationApproval returns the immutable local hosting identity for a
// canceled/expired STARTING job. It is not permission to launch. Hosting must
// fence an unclaimed attempt or reconcile an existing physical execution.
func (s *Store) CancellationApproval(ctx context.Context, id string) (out contract.Approved, err error) {
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || l.Job != id || r.Approval == nil || (!r.Job.CancelRequested && l.State != Releasing) {
			return fmt.Errorf("approved cancellation/expiry intent required")
		}
		out = *r.Approval
		return nil
	})
	return out, err
}
