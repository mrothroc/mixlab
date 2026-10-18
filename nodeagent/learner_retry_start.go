package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerjob"
)

func learnerScope(j LearnerJob, principal string, pending bool) (workload.Scope, error) {
	s, err := j.Signed.Manifest.WorkloadScope(principal)
	if err != nil {
		return s, err
	}
	if pending {
		if j.Recovery == nil {
			return s, fmt.Errorf("no authorized retry")
		}
		r := j.Recovery
		s.Controller = r.Coordinator.PrincipalID
		s.Attempt = r.Attempt.AttemptID
		s.Generation = r.Attempt.WorkloadGeneration
		s.Created = r.Created
		s.AdmitUntil = r.Expires
	} else if len(j.History) > 0 {
		r := j.History[len(j.History)-1]
		s.Controller = r.Coordinator.PrincipalID
		s.Attempt = r.Attempt.AttemptID
		s.Generation = r.Attempt.WorkloadGeneration
		s.Created = r.Created
		s.AdmitUntil = r.Expires
	}
	return s, s.Validate()
}

// StartLearnerRetry requires both node authorization and the coordinator's
// durable attempt acknowledgement. The cleanup port destroys the OLD agent
// credential before publishing a new hosting approval; an exact retry can
// replay cleanup, but may never recreate the old key or reset learner RNG.
func (s *Store) StartLearnerRetry(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, a trust.Anchor, v trust.VerifiedSnapshot, ack authenticated.Recovery, grant workload.Grant, credential workload.Result, cleanup func(context.Context, workload.Scope) error, now time.Time) (out workerjob.LearnerAssignment, err error) {
	if err = authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return out, err
	}
	if actor.Role != trust.Coordinator || cleanup == nil {
		return out, fmt.Errorf("coordinator and credential cleanup required")
	}
	if err = ack.Validate(); err != nil {
		return out, err
	}
	accepted, verifyErr := trust.VerifyProof(a, v, ack.Proof, ack.Proof.Request, now)
	if verifyErr != nil {
		return out, verifyErr
	}
	if err = workload.ValidateResult(a, grant, credential, now); err != nil {
		return out, err
	}
	if a.Cluster() != s.cluster {
		return out, fmt.Errorf("retry coordinator mismatch")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err = s.withLearnerWorkloadLock(ctx, id, func() error {
		return s.path.WithProcessLock(ctx, nodeLock, func() error {
			old, j, e := s.loadLearner(id)
			if e != nil {
				return e
			}
			m := j.Signed.Manifest
			_, ls, e := s.load()
			if e != nil {
				return e
			}
			l := findLease(&ls, m.Slot.Allocation.LeaseID)
			oldScope, replay, e := guardLearnerRetryStart(j, l, ls.Active, actor, id, ack, grant, credential, now)
			if e != nil {
				return e
			}
			if replay {
				out = *j.Assignment
				return nil
			}
			if e = cleanup(ctx, oldScope); e != nil {
				return e
			}
			if e = ctx.Err(); e != nil {
				return e
			}
			out = *j.Assignment
			out.AttemptID = ack.Body.AttemptID
			out.ExpectedPrivateHash = ack.Body.PrivateHash
			if e = out.Validate(); e != nil {
				return e
			}
			if e = s.retainLearnerOutcome(j.Outcome); e != nil {
				return e
			}
			j.History = append(j.History, *j.Recovery)
			j.RetryAcknowledgements = append(j.RetryAcknowledgements, accepted)
			j.Recovery = nil
			j.Assignment = &out
			j.Stopped = false
			j.Outcome = nil
			j.Workload = j.RetryWorkload
			j.RetryWorkload = nil
			j.Credential = &credential
			j.Version++
			return s.saveLearner(old, j)
		})
	})
	return out, err
}
