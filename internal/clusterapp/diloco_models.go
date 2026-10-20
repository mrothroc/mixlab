package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
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
	l, err := c.Release(ctx, q)
	return l, classifyLeaseReleaseError(err)
}

func classifyLeaseReleaseError(err error) error {
	var remote *NodeHTTPError
	if errors.As(err, &remote) && remote.Status == http.StatusConflict {
		return errors.Join(nodeleases.ErrReleaseConflict, err)
	}
	return err
}

// DiLoCoModels caches immutable composition and confirmed terminal cleanup.
// Run decisions are reloaded from the durable repository. Losing the cleanup
// cache only repeats release confirmation; it never skips pending cleanup.
type DiLoCoModels struct {
	Inbox      *admission.Inbox
	Principal  *principal.Store
	Repository coordinator.Repository
	Artifacts  coordinator.Artifacts
	Clock      func() time.Time
	Recover    func(context.Context, admission.Submission, coordinator.Model) (bool, error)
	cache      sync.Map
	released   sync.Map
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
func (s *DiLoCoModels) Tick(ctx context.Context, q admission.Submission, _ admission.Snapshot) (err error) {
	stage := "compose"
	defer func() { err = diLoCoStageError(stage, err) }()
	v, e := s.get(ctx, q.Launch.Run)
	if e != nil {
		return e
	}
	stage = "load run"
	state, e := s.Repository.Load(ctx, q.Launch.Run)
	if e != nil {
		return e
	}
	terminal := state.Phase == coordinator.Complete || state.Phase == coordinator.Failed
	if _, done := s.released.Load(q.Launch.Run); terminal && done {
		return nil
	}
	stage = "allocation lease maintenance"
	if terminal {
		stage = "allocation release"
	}
	released, e := v.keeper.Tick(ctx)
	if e != nil {
		state, loadErr := s.Repository.Load(ctx, q.Launch.Run)
		if loadErr != nil || state.Phase == coordinator.Complete || state.Phase == coordinator.Failed {
			return errors.Join(e, loadErr)
		}
		reason := allocationFailureReason(q, s.Clock())
		if !recoveryTransient(e) && !errors.Is(ctx.Err(), context.Canceled) {
			e = fmt.Errorf("%s: %w", reason, e)
		}
		return errors.Join((DiLoCoRecovery{Model: v.model}).failureWithReason(ctx, q.Launch.Run, state.Recovery != nil, e, reason), v.model.Tick(ctx, q.Launch.Run, int64(q.Launch.Selection.Requirements.Limits.RuntimeSeconds)))
	}
	if released {
		s.released.Store(q.Launch.Run, struct{}{})
		return nil
	}
	// The keeper can itself fail a live run. Never poll learners after that
	// transition, but keep retrying release until every node confirms it.
	stage = "load run after lease maintenance"
	state, e = s.Repository.Load(ctx, q.Launch.Run)
	if e != nil {
		return e
	}
	if state.Phase == coordinator.Complete || state.Phase == coordinator.Failed {
		return nil
	}
	if s.Recover != nil {
		_, e = s.Recover(ctx, q, v.model)
	}
	stage = "run progression"
	return errors.Join(diLoCoStageError("learner recovery", e), v.model.Tick(ctx, q.Launch.Run, int64(q.Launch.Selection.Requirements.Limits.RuntimeSeconds)))
}

func diLoCoStageError(stage string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", stage, err)
}
