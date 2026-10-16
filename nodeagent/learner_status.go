package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

// LearnerStatus exposes only operational fences, never local paths or keys.
type LearnerStatus struct {
	Binding    run.Binding              `json:"binding"`
	Allocation run.Allocation           `json:"allocation"`
	WorkerID   string                   `json:"worker_id"`
	Attempt    string                   `json:"attempt"`
	Stopped    bool                     `json:"stopped"`
	Recovery   *nodejob.LearnerRecovery `json:"recovery,omitempty"`
}

func (s *Store) LearnerStatus(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, now time.Time) (out LearnerStatus, err error) {
	if err = authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return out, err
	}
	if actor.Role != trust.Coordinator {
		return out, fmt.Errorf("coordinator required")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, j, e := s.loadLearner(id)
		if e != nil {
			return e
		}
		m := j.Signed.Manifest
		if actor.Principal != m.Plan.Coordinator.PrincipalID || j.Commit == nil || j.Assignment == nil {
			return fmt.Errorf("committed owning coordinator required")
		}
		_, ls, e := s.load()
		if e != nil {
			return e
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || ls.Active != l.ID || l.Job != id || l.State != Running || !ownsLease(actor, *l) || now.Unix() >= l.Expires {
			return fmt.Errorf("original learner allocation no longer live")
		}
		out = LearnerStatus{Binding: j.Commit.Binding, Allocation: m.Slot.Allocation, WorkerID: m.Slot.Member.WorkerID, Attempt: j.Assignment.AttemptID, Stopped: j.Stopped, Recovery: j.Recovery}
		return nil
	})
	return
}
