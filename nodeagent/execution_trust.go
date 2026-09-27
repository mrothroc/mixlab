package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

type executionCommit struct{ accepted trust.AcceptedProof }

func (j executionCommit) AcceptedGeneration(digest string) (uint64, bool, error) {
	return j.accepted.Generation, j.accepted.Digest == digest, nil
}

// CheckExecutionTrust rechecks durable admission against fresh trust. Only this
// owning context supplies historical acceptance: a caller cannot backdate a
// new job by passing a saved controller proof or an arbitrary generation.
func (s *Store) CheckExecutionTrust(ctx context.Context, anchor trust.Anchor, view trust.VerifiedSnapshot, now time.Time) error {
	if anchor.Cluster() != s.cluster {
		return fmt.Errorf("execution trust belongs to another cluster")
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, leases, err := s.load()
		if err != nil || leases.Active == "" {
			return err
		}
		lease := findLease(&leases, leases.Active)
		if lease == nil {
			return fmt.Errorf("active execution lease missing")
		}
		if lease.Job == "" {
			return nil
		}
		_, r, err := s.loadJob(lease.Job)
		if err != nil {
			return err
		}
		q, err := r.Manifest.Manifest.SigningRequest()
		if err != nil {
			return err
		}
		if err := trust.VerifyHistoricalProof(anchor, view, r.Manifest.Proof, q, executionCommit{r.Commit}, now); err != nil {
			return err
		}
		if r.Transport == nil {
			return nil
		}
		p := r.Transport.Signed
		if now.Unix() < p.Plan.Created || now.Unix() >= p.Plan.Expires {
			return fmt.Errorf("committed transport expired")
		}
		q, err = p.Plan.SigningRequest()
		if err != nil {
			return err
		}
		if err := trust.VerifyHistoricalProof(anchor, view, p.Proof, q, executionCommit{r.Transport.Commit}, now); err != nil {
			return err
		}
		for _, member := range p.Plan.Members {
			if err := trust.VerifyWorkload(anchor, view, member.Chain, member.Binding, now); err != nil {
				return err
			}
		}
		return nil
	})
}
