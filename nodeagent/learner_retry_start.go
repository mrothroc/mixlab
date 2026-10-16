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
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, e := s.loadLearner(id)
		if e != nil {
			return e
		}
		m := j.Signed.Manifest
		if a.Cluster() != s.cluster || actor.Principal != m.Plan.Coordinator.PrincipalID || ack.Coordinator != m.Plan.Coordinator {
			return fmt.Errorf("retry coordinator mismatch")
		}
		_, ls, e := s.load()
		if e != nil {
			return e
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || l.Job != id || ls.Active != l.ID || l.State != Running || !ownsLease(actor, *l) || now.Unix() >= l.Expires {
			return fmt.Errorf("retry allocation no longer live")
		}
		if j.Recovery == nil {
			if len(j.History) > 0 && j.History[len(j.History)-1].Attempt == ack.Body && j.Assignment != nil && !j.Stopped && j.Credential != nil && j.Credential.GrantHash == credential.GrantHash {
				out = *j.Assignment
				return nil
			}
			return fmt.Errorf("no pending node recovery authorization")
		}
		if j.Recovery.Attempt != ack.Body || now.Unix() >= j.Recovery.Expires || !j.Stopped || j.RetryWorkload == nil || !j.RetryWorkload.Initialized || j.Assignment == nil || j.Commit == nil || ack.Body.Binding != j.Commit.Binding {
			return fmt.Errorf("retry fence or coordinator acknowledgement mismatch")
		}
		scope, e := learnerScope(j, j.RetryWorkload.Principal, true)
		if e != nil {
			return e
		}
		if grant.Request.Request.Scope != scope {
			return fmt.Errorf("new workload credential changed retry scope")
		}
		oldScope, e := learnerScope(j, j.Workload.Principal, false)
		if e != nil {
			return e
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
		j.History = append(j.History, *j.Recovery)
		j.RetryAcknowledgements = append(j.RetryAcknowledgements, accepted)
		j.Recovery = nil
		j.Assignment = &out
		j.Stopped = false
		j.Workload = j.RetryWorkload
		j.RetryWorkload = nil
		j.Credential = &credential
		j.Version++
		return s.saveLearner(old, j)
	})
	return out, err
}
