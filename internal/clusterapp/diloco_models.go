package clusterapp

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/nodeleases"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust/principal"
)

type serviceClock struct{ now func() time.Time }

func (c serviceClock) Unix() int64 { return c.now().Unix() }

type remoteEvaluation struct{}

func (remoteEvaluation) Evaluate(context.Context, run.EvaluationAssignment) (run.EvaluationResult, error) {
	return run.EvaluationResult{}, fmt.Errorf("evaluation requires the designated learner result")
}

type serviceNodes struct {
	submission admission.Submission
	principal  *principal.Store
	clock      func() time.Time
}

func (n serviceNodes) client(a run.Allocation) (*NodeClient, error) {
	for _, c := range n.submission.Launch.Selection.Selected {
		if c.Capabilities.Node == a.NodeID {
			return NewNodeClient(n.principal, c.Endpoint, a.NodeID, n.clock)
		}
	}
	return nil, fmt.Errorf("allocation is not in accepted selection")
}
func (n serviceNodes) Status(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
	c, e := n.client(a)
	if e != nil {
		return nodeagent.Lease{}, e
	}
	return c.LeaseStatus(ctx, a.LeaseID)
}
func (n serviceNodes) Renew(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	c, e := n.client(a)
	if e != nil {
		return nodeagent.Lease{}, e
	}
	return c.Renew(ctx, q)
}
func (n serviceNodes) Release(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	c, e := n.client(a)
	if e != nil {
		return nodeagent.Lease{}, e
	}
	return c.Release(ctx, q)
}

// DiLoCoModels caches only immutable composition. Run decisions are always
// reloaded from the durable repository; losing the cache changes no semantics.
type DiLoCoModels struct {
	Inbox      *admission.Inbox
	Principal  *principal.Store
	Repository coordinator.Repository
	Artifacts  coordinator.Artifacts
	Clock      func() time.Time
	cache      sync.Map
}
type composedRun struct {
	model  coordinator.Model
	keeper nodeleases.Keeper
}

func (s *DiLoCoModels) get(ctx context.Context, id string) (composedRun, error) {
	if s.Inbox == nil || s.Principal == nil || s.Repository == nil || s.Artifacts == nil || s.Clock == nil {
		return composedRun{}, fmt.Errorf("complete coordinator composition required")
	}
	if v, ok := s.cache.Load(id); ok {
		return v.(composedRun), nil
	}
	q, err := s.Inbox.Read(ctx, id)
	if err != nil {
		return composedRun{}, err
	}
	c := serviceClock{s.Clock}
	m := coordinator.Model{Repository: s.Repository, Artifacts: s.Artifacts, Clock: c, Evaluator: remoteEvaluation{}}
	k := nodeleases.Keeper{Repository: s.Repository, Nodes: serviceNodes{q, s.Principal, s.Clock}, Clock: c, RunID: id, TTLSeconds: q.Launch.TTLSeconds}
	m.Allocations = &k
	k.Fail = m.Fail
	v, _ := s.cache.LoadOrStore(id, composedRun{m, k})
	return v.(composedRun), nil
}
func (s *DiLoCoModels) Model(ctx context.Context, id string) (coordinator.Model, error) {
	v, e := s.get(ctx, id)
	return v.model, e
}
func (s *DiLoCoModels) Tick(ctx context.Context, q admission.Submission, _ admission.Snapshot) error {
	v, e := s.get(ctx, q.Launch.Run)
	if e != nil {
		return e
	}
	if _, e = v.keeper.Tick(ctx); e != nil {
		return e
	}
	return v.model.Tick(ctx, q.Launch.Run, int64(q.Launch.Selection.Requirements.Limits.RuntimeSeconds))
}
