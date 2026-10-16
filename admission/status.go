package admission

import (
	"context"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
)

// Snapshot is an owned read of the durable saga, not authority to mutate node
// leases. Services reconstruct their effect adapters from these exact inputs.
type Snapshot struct {
	Request       recruitment.LaunchPlan `json:"request"`
	Specification Specification          `json:"specification"`
	Deadline      int64                  `json:"deadline"`
	Plan          *run.Plan              `json:"plan,omitempty"`
	Commit        *run.Commit            `json:"commit,omitempty"`
	Committed     bool                   `json:"committed"`
	Aborting      bool                   `json:"aborting"`
	Done          bool                   `json:"done"`
	Nodes         []NodeSnapshot         `json:"nodes"`
}

// NodeSnapshot exposes durable reservation and handoff evidence, not current
// lease authority. Maintenance must re-read the node before each renewal.
type NodeSnapshot struct {
	Reserve   nodeagent.Reserve `json:"reserve"`
	Touched   bool              `json:"touched"`
	Lease     *nodeagent.Lease  `json:"lease,omitempty"`
	Finalized bool              `json:"finalized"`
	Released  bool              `json:"released"`
}

func (s *Saga) Snapshot(ctx context.Context) (Snapshot, error) {
	if err := ctx.Err(); err != nil {
		return Snapshot{}, err
	}
	_, r, err := s.load()
	if err != nil {
		return Snapshot{}, err
	}
	out := Snapshot{Request: r.Request, Specification: r.Specification, Deadline: r.Deadline,
		Plan: r.Plan, Commit: r.Candidate, Committed: r.Committed, Aborting: r.Aborting, Done: r.Done}
	for _, n := range r.Nodes {
		out.Nodes = append(out.Nodes, NodeSnapshot{Reserve: n.Reserve, Touched: n.Touched, Lease: n.Lease, Finalized: n.Finalized, Released: n.Released})
	}
	return out, nil
}
