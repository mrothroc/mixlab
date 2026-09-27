package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func validateExecution(r jobRecord) error {
	j := r.Job
	if r.Transport == nil {
		if j.TransportHash != "" || r.Approval != nil {
			return fmt.Errorf("job transport evidence missing")
		}
	} else {
		q, err := r.Transport.Signed.Plan.SigningRequest()
		if err != nil {
			return err
		}
		if q.Digest != j.TransportHash || r.Transport.Signed.Proof.Request != q || r.Transport.Commit.Generation == 0 || !contract.Hash(r.Transport.Commit.Digest) {
			return fmt.Errorf("invalid committed transport evidence")
		}
		p := r.Transport.Signed.Plan
		if r.Manifest.Manifest.Rank >= len(p.Members) {
			return fmt.Errorf("transport rank missing")
		}
		local := p.Members[r.Manifest.Manifest.Rank]
		if r.Workload != nil && (!r.Workload.Initialized || local.Binding.Principal != r.Workload.Principal) {
			return fmt.Errorf("stored transport changed prepared workload identity")
		}
		if local.Job != j.ID || local.ManifestHash != j.ManifestHash || p.Attempt != j.Attempt || p.Controller != j.Controller {
			return fmt.Errorf("stored transport job binding mismatch")
		}
	}
	if r.Approval == nil {
		if j.ApprovalHash != "" || r.StartKey != "" || (r.Outcome == nil && j.State != JobPrepared) {
			return fmt.Errorf("execution approval history missing")
		}
	} else {
		if err := r.Approval.Validate(); err != nil {
			return err
		}
		if digest(*r.Approval) != j.ApprovalHash || r.Approval.ManifestHash != j.ManifestHash || r.Approval.TransportHash != j.TransportHash || r.Approval.Assignment.JobID != j.ID || r.Approval.Assignment.AttemptID != j.Attempt || !identifier(r.StartKey) {
			return fmt.Errorf("execution approval binding mismatch")
		}
	}
	if r.Outcome == nil {
		if j.OutcomeVersion != 0 {
			return fmt.Errorf("execution outcome history missing")
		}
		return nil
	}
	o := *r.Outcome
	if err := o.Validate(); err != nil {
		return err
	}
	if o.JobID != j.ID || o.AttemptID != j.Attempt || o.ManifestHash != j.ManifestHash || o.ApprovalHash != j.ApprovalHash || o.Version != j.OutcomeVersion {
		return fmt.Errorf("invalid accepted hosting outcome")
	}
	return nil
}

func findLease(r *leaseJournal, id string) *Lease {
	for i := range r.Leases {
		if r.Leases[i].ID == id {
			return &r.Leases[i]
		}
	}
	return nil
}

func (s *Store) ActivateTransport(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted grouptransport.Accepted, expectedVersion uint64, now time.Time) (out Job, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	signed, commit, id, err := accepted.Value()
	if err != nil {
		return out, err
	}
	q, err := signed.Plan.SigningRequest()
	if err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		if actor.Principal != r.Job.Controller || signed.Plan.Controller != actor.Principal || now.Unix() >= signed.Plan.Expires || now.Unix() >= r.Manifest.Manifest.Expires {
			return fmt.Errorf("transport controller/deadline mismatch")
		}
		if r.Manifest.Manifest.Rank >= len(signed.Plan.Members) {
			return fmt.Errorf("transport rank missing")
		}
		local := signed.Plan.Members[r.Manifest.Manifest.Rank]
		if r.Workload != nil && (!r.Workload.Initialized || local.Binding.Principal != r.Workload.Principal) {
			return fmt.Errorf("transport changed prepared workload identity")
		}
		if local.Job != r.Job.ID || local.Node != s.node || local.ManifestHash != r.Job.ManifestHash || signed.Plan.Attempt != r.Job.Attempt {
			return fmt.Errorf("transport changed prepared job")
		}
		if r.Transport != nil {
			if digest(r.Transport.Signed) != digest(signed) {
				return fmt.Errorf("transport already bound")
			}
			out = r.Job
			return nil
		}
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || leases.Active != l.ID || l.State != Prepared || l.Job != r.Job.ID || now.Unix() >= l.Expires || r.Job.CancelRequested || r.Job.State != JobPrepared || r.Job.Version != expectedVersion {
			return fmt.Errorf("job/lease no longer activatable")
		}
		r.Transport = &transportRecord{Signed: signed, Commit: commit}
		r.Job.TransportHash = q.Digest
		r.Job.Version++
		if err := s.saveJob(old, r); err != nil {
			return err
		}
		out = r.Job
		return nil
	})
	return out, err
}

type StartCommand struct {
	IdempotencyKey  string `json:"idempotency_key"`
	Job             string `json:"job"`
	ExpectedVersion uint64 `json:"expected_version"`
}

// StartJob accepts a locally resolved assignment, never a remote filesystem
// path. It publishes start intent before exposing an execution approval.
func (s *Store) StartJob(ctx context.Context, actor trust.AuthenticatedPrincipal, q StartCommand, a workerjob.Assignment, now time.Time) (out contract.Approved, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(q.IdempotencyKey) || !identifier(q.Job) || q.ExpectedVersion == 0 {
		return out, fmt.Errorf("invalid start command")
	}
	// Copy raw config/membership slices before entering the persistent boundary.
	b, err := json.Marshal(a)
	if err != nil {
		return out, err
	}
	var copy workerjob.Assignment
	if err := json.Unmarshal(b, &copy); err != nil {
		return out, err
	}
	a = copy
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.loadJob(q.Job)
		if err != nil {
			return err
		}
		if actor.Principal != r.Job.Controller {
			return fmt.Errorf("job belongs to another controller")
		}
		m := r.Manifest.Manifest
		v, err := distributed.NewLocalGroupView(m.Membership, m.Members[m.Rank].MemberID, m.Rank, m.Attempt)
		if err != nil {
			return err
		}
		if err := a.Validate(); err != nil {
			return err
		}
		if a.CheckpointAt != m.CheckpointAt {
			return fmt.Errorf("checkpoint stop differs from prepared job")
		}
		if len(m.Artifacts) == 0 {
			if a.Resume != nil {
				return fmt.Errorf("unapproved resume input")
			}
		} else if len(m.Artifacts) != 1 || m.Artifacts[0].Kind != "checkpoint" || a.Resume == nil || a.Resume.SHA256 != m.Artifacts[0].SHA256 || a.Resume.Bytes != m.Artifacts[0].Bytes {
			return fmt.Errorf("resume input differs from prepared job")
		}
		if a.JobID != m.Job || a.AttemptID != m.Attempt || a.BuildID != m.BuildID || digest(a.View) != digest(v) || !bytes.Equal(a.Config, m.Config) || a.DatasetSelector != m.DatasetSelector || a.DatasetSHA256 != m.DatasetID || a.ProgramSHA256 != m.ProgramHash || a.RuntimeSeconds != m.Limits.RuntimeSeconds || a.WeightLayoutSHA256 != m.WeightLayoutHash || a.OptimizerSHA256 != m.OptimizerHash {
			return fmt.Errorf("local assignment differs from prepared job")
		}
		approved := contract.Approved{Version: contract.Version, ManifestHash: r.Job.ManifestHash, TransportHash: r.Job.TransportHash, Assignment: a, Limits: contract.Limits{CPUSeconds: m.Limits.CPUSeconds, MemoryBytes: int64(m.Limits.MemoryBytes), DiskBytes: int64(m.Limits.DiskBytes), LogBytes: int64(m.Limits.LogBytes)}}
		if err := approved.Validate(); err != nil {
			return err
		}
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || leases.Active != l.ID || l.State != Prepared || l.Job != r.Job.ID || now.Unix() >= l.Expires || now.Unix() >= m.Expires || r.Job.CancelRequested || r.Outcome != nil || r.Transport == nil || now.Unix() >= r.Transport.Signed.Plan.Expires {
			return fmt.Errorf("job/lease/transport no longer startable; reconcile execution")
		}
		if r.Approval != nil {
			if r.StartKey != q.IdempotencyKey || digest(approved) != r.Job.ApprovalHash {
				return fmt.Errorf("start retry changed approval/key")
			}
			out = *r.Approval
			return nil
		}
		if r.Job.State != JobPrepared || r.Job.Version != q.ExpectedVersion {
			return fmt.Errorf("job/lease/transport no longer startable")
		}
		r.Approval = &approved
		r.StartKey = q.IdempotencyKey
		r.Job.ApprovalHash = digest(approved)
		r.Job.State = JobStarting
		r.Job.Version++
		if err := s.saveJob(old, r); err != nil {
			return err
		}
		out = approved
		return nil
	})
	return out, err
}

// ApplyOutcome is called only by the local worker-host outcome port. No remote
// endpoint may synthesize these confirmations. Publication is job-first and
// lease-second; replay repairs an interruption between those writes.
func (s *Store) ApplyOutcome(ctx context.Context, o contract.Outcome) (out Job, err error) {
	if err := o.Validate(); err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.loadJob(o.JobID)
		if err != nil {
			return err
		}
		if r.Job.Attempt != o.AttemptID || r.Job.ManifestHash != o.ManifestHash || r.Job.ApprovalHash != o.ApprovalHash {
			return fmt.Errorf("unapproved execution outcome")
		}
		leaseBytes, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || l.Job != r.Job.ID {
			return fmt.Errorf("outcome lease binding missing")
		}
		if r.Approval == nil && (!o.NoChild || o.Kind != contract.Canceled || o.PID != 0 || (!r.Job.CancelRequested && l.State != Releasing && l.State != Released)) {
			return fmt.Errorf("unrequested preparation fence outcome")
		}
		if r.Outcome != nil {
			if o.Version < r.Outcome.Version {
				return fmt.Errorf("outcome rollback")
			}
			if o.Version == r.Outcome.Version && o != *r.Outcome {
				return fmt.Errorf("outcome version conflict")
			}
			if r.Outcome.Terminal() && o != *r.Outcome {
				return fmt.Errorf("terminal outcome is immutable")
			}
		}
		if r.Outcome == nil || *r.Outcome != o {
			r.Outcome = &o
			r.Job.OutcomeVersion = o.Version
			r.Job.Version++
			switch o.Kind {
			case contract.Started:
				r.Job.State = JobRunning
			case contract.Exited:
				r.Job.State = JobExited
			case contract.Canceled:
				r.Job.State = JobCanceled
				if r.Approval == nil && l.CleanupReason == "expired" {
					r.Job.State = JobExpired
				}
			case contract.Failed:
				r.Job.State = JobFailed
			}
			if err := s.saveJob(old, r); err != nil {
				return err
			}
		}
		changed := false
		if o.Terminal() {
			if l.State != Releasing && l.State != Released {
				l.State = Releasing
				l.CleanupReason = o.Kind
				changed = true
			}
		} else if l.State == Prepared {
			l.State = Running
			changed = true
		}
		if changed {
			l.Version++
			leases.NodeVersion++
			if err := s.save(leaseBytes, leases); err != nil {
				return err
			}
		}
		out = r.Job
		return nil
	})
	return out, err
}

type CleanupRequest struct {
	Job, Attempt, TransportHash string
	Workload                    workload.Scope
	WorkloadInitialized         bool
}

func cleanupRequest(r jobRecord, cluster string) (CleanupRequest, error) {
	q := CleanupRequest{Job: r.Job.ID, Attempt: r.Job.Attempt, TransportHash: r.Job.TransportHash}
	if r.Workload != nil {
		var err error
		q.Workload, err = r.Manifest.Manifest.WorkloadScope(cluster, r.Workload.Principal)
		if err != nil {
			return CleanupRequest{}, err
		}
		q.WorkloadInitialized = r.Workload.Initialized
	}
	return q, nil
}

// CompleteCleanup releases the accelerator only after terminal hosting evidence
// and the owning credential/transport cleanup ports succeed. Cleanup must be
// idempotent; an interrupted publication replays it before freeing the lease.
func (s *Store) CompleteCleanup(ctx context.Context, id string, cleanup func(context.Context, CleanupRequest) error) error {
	if cleanup == nil {
		return fmt.Errorf("cleanup port required")
	}
	var request CleanupRequest
	var outcome contract.Outcome
	alreadyReleased := false
	if err := s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		if r.Outcome == nil || !r.Outcome.Terminal() || !r.Outcome.NoChild {
			return fmt.Errorf("terminal child cleanup unconfirmed")
		}
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || l.Job != id {
			return fmt.Errorf("cleanup lease missing")
		}
		if l.State == Released {
			alreadyReleased = true
			return nil
		}
		if l.State != Releasing || leases.Active != l.ID {
			return fmt.Errorf("terminal outcome must be reconciled before release")
		}
		request, err = cleanupRequest(r, s.cluster)
		if err != nil {
			return err
		}
		outcome = *r.Outcome
		return nil
	}); err != nil {
		return err
	}
	if alreadyReleased {
		return nil
	}
	// RELEASING is the durable cleanup intent. External ports run without the
	// node lock; exact terminal evidence is revalidated before publication.
	cleanupCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	if err := cleanup(cleanupCtx, request); err != nil {
		return err
	}
	if err := cleanupCtx.Err(); err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		current, err := cleanupRequest(r, s.cluster)
		if err != nil {
			return err
		}
		if r.Outcome == nil || *r.Outcome != outcome || request != current {
			return fmt.Errorf("cleanup job evidence changed")
		}
		old, leases, err := s.load()
		if err != nil {
			return err
		}
		l := findLease(&leases, r.Job.Lease)
		if l == nil || l.Job != id {
			return fmt.Errorf("cleanup lease missing")
		}
		if l.State == Released {
			return nil
		}
		if l.State != Releasing || leases.Active != l.ID {
			return fmt.Errorf("cleanup lease changed")
		}
		l.State = Released
		l.Version++
		leases.Active = ""
		leases.NodeVersion++
		return s.save(old, leases)
	})
}
