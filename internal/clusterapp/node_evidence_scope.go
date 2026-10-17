package clusterapp

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

// Preparation can precede any job journal. Check the owning reservation before
// fetching authority state; the store repeats admission checks at mutation.
func (a *NodeJobs) preparationScope(ctx context.Context, actor trust.AuthenticatedPrincipal, lease, run, job, kind string, version uint64) error {
	l, err := a.options.Store.LeaseStatus(ctx, actor, lease, a.options.Clock())
	if err != nil {
		return err
	}
	x, err := a.options.Store.ActiveExecution(ctx)
	if err != nil {
		return err
	}
	if x == nil || x.Lease.ID != lease || l.Controller != actor.Principal || l.Run != run || a.options.Clock().Unix() >= l.Expires {
		return fmt.Errorf("live original reservation required")
	}
	if l.State == nodeagent.Prepared && l.Job == job && l.JobKind == kind {
		return nil
	}
	if l.State != nodeagent.Reserved || l.Job != "" || l.Version != version {
		return fmt.Errorf("reservation scope or version mismatch")
	}
	return nil
}
