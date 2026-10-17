package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Preflight checks local ownership and scope before adapters fetch authority
// state. It grants no permission: mutations repeat the same guards under lock.
func (s *Store) checkLearnerEvidence(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, now time.Time, check func(LearnerJob, *Lease, string) error) error {
	if err := authorizeLeaseActor(actor, s.cluster, now); err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, j, err := s.loadLearner(id)
		if err != nil {
			return err
		}
		_, ls, err := s.load()
		if err != nil {
			return err
		}
		return check(j, findLease(&ls, j.Signed.Manifest.Slot.Allocation.LeaseID), ls.Active)
	})
}

func (s *Store) CheckLearnerCredential(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, g workload.Grant, result workload.Result, now time.Time) error {
	return s.checkLearnerEvidence(ctx, actor, id, now, func(j LearnerJob, l *Lease, active string) error {
		return guardLearnerCredential(j, l, active, actor, id, g, result, now)
	})
}

func guardLearnerCredential(j LearnerJob, l *Lease, active string, actor trust.AuthenticatedPrincipal, id string, g workload.Grant, result workload.Result, now time.Time) error {
	m := j.Signed.Manifest
	if actor.Role != trust.Controller || j.Workload == nil || !j.Workload.Initialized || j.Stopped || m.Controller != actor.Principal {
		return fmt.Errorf("prepared learner credential required")
	}
	scope, err := m.WorkloadScope(j.Workload.Principal)
	if err != nil {
		return err
	}
	if g.Request.Request.Scope != scope {
		return fmt.Errorf("credential scope differs from learner")
	}
	if l == nil || l.ID != active || l.State != Prepared || l.Job != id || now.Unix() >= l.Expires || now.Unix() >= m.Plan.ExpiresAt {
		return fmt.Errorf("learner lease expired")
	}
	if j.Credential != nil && (j.Credential.Binding != result.Binding || j.Credential.GrantHash != result.GrantHash) {
		return fmt.Errorf("credential changed")
	}
	return nil
}

func (s *Store) CheckLearnerCommit(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, c authenticated.Commit, now time.Time) error {
	return s.checkLearnerEvidence(ctx, actor, id, now, func(j LearnerJob, l *Lease, active string) error {
		return guardLearnerCommit(j, l, active, actor, id, c)
	})
}

func guardLearnerCommit(j LearnerJob, l *Lease, active string, actor trust.AuthenticatedPrincipal, id string, c authenticated.Commit) error {
	m := j.Signed.Manifest
	controller := actor.Role == trust.Controller && actor.Principal == m.Controller
	coordinator := actor.Role == trust.Coordinator && actor.Principal == m.Plan.Coordinator.PrincipalID
	if !controller && !coordinator {
		return fmt.Errorf("handoff actor mismatch")
	}
	if err := c.Body.ValidateFor(m.Plan); err != nil {
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
	if l == nil || (l.State != Released && l.ID != active) || l.Job != id || l.JobKind != "learner" || (l.State != Prepared && l.State != Running && l.State != Releasing && l.State != Released) {
		return fmt.Errorf("original learner allocation no longer live")
	}
	if j.Commit != nil && j.Commit.Binding != c.Body.Binding {
		return fmt.Errorf("conflicting run commit")
	}
	return nil
}

func (s *Store) CheckLearnerRetryStart(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, ack authenticated.Recovery, grant workload.Grant, credential workload.Result, now time.Time) error {
	return s.checkLearnerEvidence(ctx, actor, id, now, func(j LearnerJob, l *Lease, active string) error {
		_, _, err := guardLearnerRetryStart(j, l, active, actor, id, ack, grant, credential, now)
		return err
	})
}

func guardLearnerRetryStart(j LearnerJob, l *Lease, active string, actor trust.AuthenticatedPrincipal, id string, ack authenticated.Recovery, grant workload.Grant, credential workload.Result, now time.Time) (old workload.Scope, replay bool, err error) {
	m := j.Signed.Manifest
	if actor.Role != trust.Coordinator || actor.Principal != m.Plan.Coordinator.PrincipalID || ack.Coordinator != m.Plan.Coordinator {
		return old, false, fmt.Errorf("retry coordinator mismatch")
	}
	if l == nil || l.Job != id || active != l.ID || l.State != Running || !ownsLease(actor, *l) || now.Unix() >= l.Expires {
		return old, false, fmt.Errorf("retry allocation no longer live")
	}
	if j.Recovery == nil {
		if len(j.History) > 0 && j.History[len(j.History)-1].Attempt == ack.Body && j.Assignment != nil && !j.Stopped && j.Credential != nil && j.Credential.GrantHash == credential.GrantHash {
			return old, true, nil
		}
		return old, false, fmt.Errorf("no pending node recovery authorization")
	}
	if j.Recovery.Attempt != ack.Body || now.Unix() >= j.Recovery.Expires || !j.Stopped || j.RetryWorkload == nil || !j.RetryWorkload.Initialized || j.Assignment == nil || j.Commit == nil || ack.Body.Binding != j.Commit.Binding {
		return old, false, fmt.Errorf("retry fence or coordinator acknowledgement mismatch")
	}
	scope, err := learnerScope(j, j.RetryWorkload.Principal, true)
	if err != nil {
		return old, false, err
	}
	if grant.Request.Request.Scope != scope {
		return old, false, fmt.Errorf("new workload credential changed retry scope")
	}
	old, err = learnerScope(j, j.Workload.Principal, false)
	return old, false, err
}
