package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

// AuthorizeLearnerRetry records the node's explicit same-allocation decision.
// The caller supplies the coordinator's expected private snapshot digest; the
// coordinator independently compares it to its own durable round head. This
// method does not launch a child or mint credentials.
func (s *Store) AuthorizeLearnerRetry(ctx context.Context, actor trust.AuthenticatedPrincipal, id, next, private string, now time.Time) (out nodejob.LearnerRecovery, err error) {
	if err = authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return out, err
	}
	if actor.Role != trust.Coordinator || !identifier(next) {
		return out, fmt.Errorf("coordinator and fresh attempt ID required")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, e := s.loadLearner(id)
		if e != nil {
			return e
		}
		m := j.Signed.Manifest
		if j.Commit == nil || j.Assignment == nil || !j.Stopped || actor.Principal != m.Plan.Coordinator.PrincipalID {
			return fmt.Errorf("committed stopped learner required")
		}
		_, ls, e := s.load()
		if e != nil {
			return e
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || ls.Active != l.ID || l.Job != id || l.State != Running || !ownsLease(actor, *l) || now.Unix() >= l.Expires {
			return fmt.Errorf("original learner allocation expired")
		}
		if j.Recovery != nil {
			if j.Recovery.Attempt.AttemptID != next || j.Recovery.Attempt.PrivateHash != private || now.Unix() >= j.Recovery.Expires {
				return fmt.Errorf("retry already authorized or expired")
			}
			out = *j.Recovery
			return nil
		}
		var reg run.Registration
		for _, r := range j.Commit.Registrations {
			if r.Slot == m.Slot {
				reg = r
			}
		}
		for _, prior := range j.History {
			if next == prior.Attempt.PreviousAttemptID || next == prior.Attempt.AttemptID {
				return fmt.Errorf("attempt ID cannot be reused")
			}
			reg.WorkloadGeneration = prior.Attempt.WorkloadGeneration
		}
		if len(j.History) >= 1024 {
			return fmt.Errorf("retry history capacity reached")
		}
		if reg.WorkloadGeneration == ^uint64(0) {
			return fmt.Errorf("workload generation overflow")
		}
		out = nodejob.LearnerRecovery{Coordinator: m.Plan.Coordinator, Created: now.Unix(), Expires: min(now.Unix()+600, l.Expires, l.WorkloadDeadline-1), Attempt: run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: j.Commit.Binding, WorkerID: m.Slot.Member.WorkerID, PreviousAttemptID: j.Assignment.AttemptID, AttemptID: next, WorkloadGeneration: reg.WorkloadGeneration + 1, Allocation: m.Slot.Allocation, PrivateHash: private, PrecisionHash: reg.PrecisionHash}}
		out.Attempt.NodeAuthorizationHash, e = out.Digest()
		if e != nil {
			return e
		}
		if e = out.Validate(); e != nil {
			return e
		}
		if e = out.Attempt.ValidateFor(*j.Commit, j.Assignment.AttemptID, reg.WorkloadGeneration, private); e != nil {
			return e
		}
		j.Recovery = &out
		j.Version++
		return s.saveLearner(old, j)
	})
	return out, err
}
