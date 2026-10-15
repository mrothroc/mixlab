package clusterapp

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

// LearnerAllocations is the coordinator's pinned-node transport adapter. The
// client resolver uses coordinator-owned current credentials; there is no
// submitting-terminal callback or reservation/replacement operation here.
type LearnerAllocations struct {
	Client func(nodeID string) (*NodeClient, error)
}

func (a LearnerAllocations) client(allocation run.Allocation) (*NodeClient, error) {
	if err := allocation.Validate(); err != nil {
		return nil, err
	}
	if a.Client == nil {
		return nil, fmt.Errorf("coordinator node client resolver required")
	}
	c, err := a.Client(allocation.NodeID)
	if err != nil {
		return nil, err
	}
	if c == nil || c.node != allocation.NodeID {
		return nil, fmt.Errorf("allocation client pinned to another node")
	}
	return c, nil
}
func (a LearnerAllocations) Status(ctx context.Context, allocation run.Allocation) (nodeagent.Lease, error) {
	c, err := a.client(allocation)
	if err != nil {
		return nodeagent.Lease{}, err
	}
	return c.LeaseStatus(ctx, allocation.LeaseID)
}
func (a LearnerAllocations) Renew(ctx context.Context, allocation run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	c, err := a.client(allocation)
	if err != nil {
		return nodeagent.Lease{}, err
	}
	if q.Lease != allocation.LeaseID {
		return nodeagent.Lease{}, fmt.Errorf("renew changed allocation")
	}
	return c.Renew(ctx, q)
}
func (a LearnerAllocations) Release(ctx context.Context, allocation run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	c, err := a.client(allocation)
	if err != nil {
		return nodeagent.Lease{}, err
	}
	if q.Lease != allocation.LeaseID {
		return nodeagent.Lease{}, fmt.Errorf("release changed allocation")
	}
	return c.Release(ctx, q)
}
