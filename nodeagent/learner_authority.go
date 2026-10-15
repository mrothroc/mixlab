package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

func (s *Store) guardLearnerHandoff(l Lease, actor trust.AuthenticatedPrincipal) error {
	if l.JobKind != "learner" || l.Job == "" {
		return nil
	}
	_, j, err := s.loadLearner(l.Job)
	if err != nil {
		return err
	}
	if j.Commit != nil && (actor.Role != trust.Coordinator || actor.Principal != j.Commit.Coordinator.PrincipalID) {
		return fmt.Errorf("run commit has fenced submitting controller")
	}
	return nil
}

// ActivateLearnerCredential verifies issuance and binds the public result. The
// private transport key remains in the agent's separate credential context.
func (s *Store) ActivateLearnerCredential(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, a trust.Anchor, g workload.Grant, result workload.Result, now time.Time) error {
	if err := authorize(actor, s.cluster, now); err != nil {
		return err
	}
	if err := workload.ValidateResult(a, g, result, now); err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, err := s.loadLearner(id)
		if err != nil {
			return err
		}
		m := j.Signed.Manifest
		if j.Workload == nil || !j.Workload.Initialized || j.Stopped || m.Controller != actor.Principal {
			return fmt.Errorf("prepared learner credential required")
		}
		scope, err := m.WorkloadScope(j.Workload.Principal)
		if err != nil {
			return err
		}
		if g.Request.Request.Scope != scope || a.Cluster() != s.cluster {
			return fmt.Errorf("credential scope differs from learner")
		}
		_, ls, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || l.ID != ls.Active || l.State != Prepared || l.Job != id || now.Unix() >= l.Expires || now.Unix() >= m.Plan.ExpiresAt {
			return fmt.Errorf("learner lease expired")
		}
		if j.Credential != nil {
			if j.Credential.Binding != result.Binding || j.Credential.GrantHash != result.GrantHash {
				return fmt.Errorf("credential changed")
			}
			return nil
		}
		j.Credential = &result
		j.Version++
		return s.saveLearner(old, j)
	})
}

// FinalizeLearner transfers renewal/release authority after a verified commit.
// Job evidence is written first; replay repairs the lease publication. A round
// cannot start until every allocation reports this handoff as complete.
func (s *Store) FinalizeLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, a trust.Anchor, v trust.VerifiedSnapshot, c authenticated.Commit, now time.Time) error {
	if err := authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return err
	}
	if err := c.Validate(); err != nil {
		return err
	}
	if a.Cluster() != s.cluster {
		return fmt.Errorf("wrong cluster")
	}
	proof, err := trust.VerifyProof(a, v, c.Proof, c.Proof.Request, now)
	if err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, err := s.loadLearner(id)
		if err != nil {
			return err
		}
		m := j.Signed.Manifest
		controller := actor.Role == trust.Controller && actor.Principal == m.Controller
		coordinator := actor.Role == trust.Coordinator && actor.Principal == m.Plan.Coordinator.PrincipalID
		if !controller && !coordinator {
			return fmt.Errorf("handoff actor mismatch")
		}
		if err = c.Body.ValidateFor(m.Plan); err != nil {
			return err
		}
		if j.Assignment == nil || j.Credential == nil {
			return fmt.Errorf("learner is not started")
		}
		found := false
		for _, r := range c.Body.Registrations {
			if r.Slot == m.Slot {
				q, _ := m.SigningRequest()
				found = r.AttemptID == m.Attempt && r.WorkloadGeneration == m.WorkloadGeneration && r.JobManifestHash == q.Digest
			}
		}
		if !found {
			return fmt.Errorf("commit changed prepared learner")
		}
		lb, ls, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || (l.State != Released && l.ID != ls.Active) || l.Job != id || l.JobKind != "learner" || (l.State != Prepared && l.State != Running && l.State != Releasing && l.State != Released) {
			return fmt.Errorf("original learner allocation no longer live")
		}
		if j.Commit != nil {
			if j.Commit.Binding != c.Body.Binding {
				return fmt.Errorf("conflicting run commit")
			}
		} else {
			j.Commit = &c.Body
			j.CommitProof = proof
			j.Version++
			if err = s.saveLearner(old, j); err != nil {
				return err
			}
		}
		if l.Coordinator != "" {
			if l.Coordinator != c.Body.Coordinator.PrincipalID || l.RunCommitHash != c.Body.Binding.CommitHash {
				return fmt.Errorf("lease authority mismatch")
			}
			return nil
		}
		deadline, err := m.WorkloadDeadline()
		if err != nil {
			return err
		}
		l.Coordinator = c.Body.Coordinator.PrincipalID
		l.RunCommitHash = c.Body.Binding.CommitHash
		l.WorkloadDeadline = deadline.Unix()
		l.Expires = min(l.Expires, l.WorkloadDeadline)
		l.RenewBy = min(l.RenewBy, l.Expires)
		// A commit can win just before process failure or lease expiry. Adopt
		// cleanup authority as well, without reviving or extending the lease.
		if l.State != Released {
			if j.Stopped || now.Unix() >= l.Expires || l.State == Releasing {
				l.State = Releasing
				if l.CleanupReason == "" {
					l.CleanupReason = "failed"
					if now.Unix() >= l.Expires {
						l.CleanupReason = "expired"
					}
				}
			} else {
				l.State = Running
			}
		}
		l.Version++
		ls.NodeVersion++
		return s.save(lb, ls)
	})
}

// CleanupLearner holds the accelerator until hosting and credential owners
// confirm cleanup. Unstarted preparations need no process termination proof.
func (s *Store) CleanupLearner(ctx context.Context, id string, cleanup func(context.Context, workload.Scope) error) error {
	if cleanup == nil {
		return fmt.Errorf("credential cleanup port required")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, j, err := s.loadLearner(id)
		if err != nil {
			return err
		}
		m := j.Signed.Manifest
		old, ls, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if l == nil || l.Job != id {
			return fmt.Errorf("learner allocation missing")
		}
		if l.State == Released {
			return nil
		}
		if l.State != Releasing || j.Assignment != nil && !j.Stopped {
			return fmt.Errorf("release intent and confirmed stopped child required")
		}
		if j.Workload != nil {
			scope, err := learnerScope(j, j.Workload.Principal, false)
			if err != nil {
				return err
			}
			if err = cleanup(ctx, scope); err != nil {
				return err
			}
		}
		if j.RetryWorkload != nil {
			scope, err := learnerScope(j, j.RetryWorkload.Principal, true)
			if err != nil {
				return err
			}
			if err = cleanup(ctx, scope); err != nil {
				return err
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		l.State = Released
		l.Version++
		ls.Active = ""
		ls.NodeVersion++
		return s.save(old, ls)
	})
}
