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
	if a.Cluster() != s.cluster {
		return fmt.Errorf("credential scope differs from learner")
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, err := s.loadLearner(id)
		if err != nil {
			return err
		}
		m := j.Signed.Manifest
		_, ls, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if err := guardLearnerCredential(j, l, ls.Active, actor, id, g, result, now); err != nil {
			return err
		}
		if j.Credential != nil {
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
		lb, ls, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&ls, m.Slot.Allocation.LeaseID)
		if err := guardLearnerCommit(j, l, ls.Active, actor, id, c); err != nil {
			return err
		}
		if j.Commit != nil {
			// An authenticated exact commit replay upgrades pre-hosting journals
			// without inventing missing proof or changing their authority.
			if j.SignedCommitProof == nil {
				j.SignedCommitProof = &c.Proof
				j.CommitProof = proof
				j.Version++
				if err = s.saveLearner(old, j); err != nil {
					return err
				}
			}
		} else {
			j.Commit = &c.Body
			j.CommitProof = proof
			j.SignedCommitProof = &c.Proof
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

// LearnerCleanupTimeout bounds one cleanup attempt, including lifecycle locks.
const LearnerCleanupTimeout = 50 * time.Second

// CleanupLearner holds the accelerator until hosting and credential owners
// confirm cleanup. Async preparation owners must be canceled/joined BEFORE this
// call, never in cleanup: the callback holds the job's credential lifecycle lock.
// Unstarted preparations need no process termination proof.
func (s *Store) CleanupLearner(ctx context.Context, id string, cleanup func(context.Context, CleanupRequest) error) error {
	if cleanup == nil {
		return fmt.Errorf("credential cleanup port required")
	}
	ctx, cancel := context.WithTimeout(ctx, LearnerCleanupTimeout)
	defer cancel()
	return s.withLearnerWorkloadLock(ctx, id, func() error {
		var requests []CleanupRequest
		var released bool
		visit := func(finish bool) error {
			return s.path.WithProcessLock(ctx, nodeLock, func() error {
				_, j, err := s.loadLearner(id)
				if err != nil {
					return err
				}
				old, ls, err := s.load()
				if err != nil {
					return err
				}
				l := findLease(&ls, j.Signed.Manifest.Slot.Allocation.LeaseID)
				if l == nil || l.Job != id {
					return fmt.Errorf("learner allocation missing")
				}
				if l.State == Released {
					released = true
					return nil
				}
				if l.State != Releasing || j.Assignment != nil && !j.Stopped {
					return fmt.Errorf("release intent and confirmed stopped child required")
				}
				var current []CleanupRequest
				for _, retry := range []bool{false, true} {
					if i := *learnerWorkloadIntent(&j, retry); i != nil {
						scope, err := learnerScope(j, i.Principal, retry)
						if err != nil {
							return err
						}
						current = append(current, CleanupRequest{Job: id, Attempt: scope.Attempt, Workload: scope, WorkloadInitialized: i.Initialized})
					}
				}
				if !finish {
					requests = current
					return nil
				}
				if digest(current) != digest(requests) {
					return fmt.Errorf("learner cleanup scope changed")
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
		if err := visit(false); err != nil || released {
			return err
		}
		for _, request := range requests {
			if err := cleanup(ctx, request); err != nil {
				return err
			}
		}
		return visit(true)
	})
}
