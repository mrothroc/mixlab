package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func learnerApproval(j LearnerJob) (*contract.Approved, error) {
	if j.Assignment == nil {
		return nil, nil
	}
	m := j.Signed.Manifest
	q, err := m.SigningRequest()
	if err != nil {
		return nil, err
	}
	a := &contract.Approved{Version: contract.LearnerVersion, ManifestHash: q.Digest, TransportHash: m.Plan.SHA256, Learner: j.Assignment,
		Limits: contract.Limits{CPUSeconds: m.Limits.CPUSeconds, MemoryBytes: int64(m.Limits.MemoryBytes), DiskBytes: int64(m.Limits.DiskBytes), LogBytes: int64(m.Limits.LogBytes)}}
	return a, a.Validate()
}

func validateLearnerHosting(j LearnerJob) error {
	if j.SignedCommitProof != nil {
		if j.Commit == nil {
			return fmt.Errorf("commit proof without commit")
		}
		if err := (authenticated.Commit{Body: *j.Commit, Proof: *j.SignedCommitProof}).Validate(); err != nil {
			return err
		}
	}
	if j.Outcome == nil {
		return nil
	}
	a, err := learnerApproval(j)
	if err != nil {
		return err
	}
	if a == nil {
		return fmt.Errorf("learner hosting outcome without approval")
	}
	o := *j.Outcome
	if err := o.Validate(); err != nil {
		return err
	}
	if o.JobID != a.JobID() || o.AttemptID != a.AttemptID() || o.ManifestHash != a.ManifestHash || o.ApprovalHash != digest(*a) || (o.Terminal() && !j.Stopped) {
		return fmt.Errorf("learner hosting outcome binding mismatch")
	}
	return nil
}

// The generic job view carries only lifecycle identity, not a DDP membership.
// Learner assignment, handoff and recovery remain owned by the learner variant.
func learnerExecution(l Lease, j LearnerJob) (*LocalExecution, error) {
	m := j.Signed.Manifest
	if l.Job != m.Slot.Allocation.JobID || l.ID != m.Slot.Allocation.LeaseID || l.Controller != m.Controller {
		return nil, fmt.Errorf("learner execution ownership mismatch")
	}
	q, err := m.SigningRequest()
	if err != nil {
		return nil, err
	}
	x := &LocalExecution{Lease: l, Learner: &j, Outcome: j.Outcome, Job: &Job{Format: JobVersion, ID: l.Job, Attempt: m.Attempt, Lease: l.ID, Controller: m.Controller, ManifestHash: q.Digest, Version: j.Version, State: JobPrepared}}
	x.Approval, err = learnerApproval(j)
	if err != nil {
		return nil, err
	}
	if x.Approval != nil {
		x.Job.Attempt, x.Job.ApprovalHash, x.Job.State = x.Approval.AttemptID(), digest(*x.Approval), JobStarting
	}
	if j.Workload != nil && j.Workload.Initialized {
		x.Workload, err = learnerScope(j, j.Workload.Principal, false)
		if err != nil {
			return nil, err
		}
	}
	if j.Outcome != nil {
		x.Job.OutcomeVersion = j.Outcome.Version
		switch j.Outcome.Kind {
		case contract.Started:
			x.Job.State = JobRunning
		case contract.Exited:
			x.Job.State = JobExited
		case contract.Failed:
			x.Job.State = JobFailed
		case contract.Canceled:
			x.Job.State = JobCanceled
		}
	}
	return x, nil
}

// ApplyLearnerOutcome records physical hosting evidence without surrendering
// the original allocation. Only an explicit release/expiry frees it; a failed
// child can be retried under P5's fenced same-worker recovery protocol.
func (s *Store) ApplyLearnerOutcome(ctx context.Context, o contract.Outcome) error {
	if err := o.Validate(); err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, err := s.loadLearner(o.JobID)
		if err != nil {
			return err
		}
		a, err := learnerApproval(j)
		if err != nil {
			return err
		}
		if a == nil || o.AttemptID != a.AttemptID() || o.ManifestHash != a.ManifestHash || o.ApprovalHash != digest(*a) {
			return fmt.Errorf("unapproved learner outcome")
		}
		if j.Outcome != nil {
			if o == *j.Outcome {
				return nil
			}
			if j.Outcome.Terminal() || o.Version <= j.Outcome.Version {
				return fmt.Errorf("learner outcome rollback or conflict")
			}
		}
		if err := s.retainLearnerOutcome(j.Outcome); err != nil {
			return err
		}
		if err := s.retainLearnerOutcome(&o); err != nil {
			return err
		}
		j.Outcome = &o
		if o.Terminal() {
			j.Stopped = true
		}
		j.Version++
		return s.saveLearner(old, j)
	})
}

func checkLearnerExecutionTrust(anchor trust.Anchor, view trust.VerifiedSnapshot, j LearnerJob, now time.Time) error {
	m := j.Signed.Manifest
	q, err := m.SigningRequest()
	if err != nil {
		return err
	}
	if err = trust.VerifyHistoricalProof(anchor, view, j.Signed.Proof, q, executionCommit{j.Accepted}, now); err != nil {
		return err
	}
	deadline, err := m.WorkloadDeadline()
	if err != nil {
		return err
	}
	if !now.Before(deadline) || now.Unix() < m.Plan.CreatedAt {
		return fmt.Errorf("learner workload expired")
	}
	if j.Commit == nil && now.Unix() >= m.Plan.ExpiresAt {
		return fmt.Errorf("uncommitted learner admission expired")
	}
	if j.Commit != nil {
		if j.SignedCommitProof == nil {
			return fmt.Errorf("learner commit lacks historical proof; reconcile before execution")
		}
		p := *j.SignedCommitProof
		if err = trust.VerifyHistoricalProof(anchor, view, p, p.Request, executionCommit{j.CommitProof}, now); err != nil {
			return err
		}
	}
	if j.Credential != nil {
		return trust.VerifyWorkload(anchor, view, j.Credential.Chain, j.Credential.Binding, now)
	}
	return nil
}
