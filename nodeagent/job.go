package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

const JobVersion = "mixlab_node_job_state_v1"
const (
	JobPrepared = "PREPARED"
	JobStarting = "STARTING"
	JobRunning  = "RUNNING"
	JobExited   = "EXITED"
	JobFailed   = "FAILED"
	JobCanceled = "CANCELED"
	JobExpired  = "EXPIRED"
)

type Job struct {
	Format          string `json:"format"`
	ID              string `json:"id"`
	Attempt         string `json:"attempt"`
	Lease           string `json:"lease"`
	Controller      string `json:"controller"`
	ManifestHash    string `json:"manifest_hash"`
	Version         uint64 `json:"version"`
	State           string `json:"state"`
	CancelRequested bool   `json:"cancel_requested"`
	TransportHash   string `json:"transport_hash"`
	OutcomeVersion  uint64 `json:"outcome_version"`
	ApprovalHash    string `json:"approval_hash"`
}
type jobRecord struct {
	Job       Job                 `json:"job"`
	Manifest  nodejob.Signed      `json:"manifest"`
	Commit    trust.AcceptedProof `json:"commit"`
	Transport *transportRecord    `json:"transport"`
	Approval  *contract.Approved  `json:"approval"`
	Outcome   *contract.Outcome   `json:"outcome"`
	StartKey  string              `json:"start_key"`
	Workload  *workloadIntent     `json:"workload,omitempty"`
}
type transportRecord struct {
	Signed grouptransport.Signed `json:"signed"`
	Commit trust.AcceptedProof   `json:"commit"`
}

func jobFile(id string) (string, error) {
	if !identifier(id) {
		return "", fmt.Errorf("invalid job ID")
	}
	return "job-" + id + ".json", nil
}
func (s *Store) loadJob(id string) ([]byte, jobRecord, error) {
	file, err := jobFile(id)
	if err != nil {
		return nil, jobRecord{}, err
	}
	b, err := s.path.ReadFileLimit(file, 1<<20)
	if err != nil {
		return nil, jobRecord{}, err
	}
	var r jobRecord
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) {
		return nil, r, fmt.Errorf("noncanonical node job")
	}
	if err := s.validateJob(r); err != nil {
		return nil, r, err
	}
	return b, r, nil
}
func (s *Store) validateJob(r jobRecord) error {
	j, m := r.Job, r.Manifest.Manifest
	request, err := m.SigningRequest()
	if err != nil {
		return err
	}
	if j.Format != JobVersion || j.ID != m.Job || j.Attempt != m.Attempt || j.Lease != m.Lease || j.Controller != m.Controller || j.ManifestHash != request.Digest || j.Version == 0 || m.Node != s.node || r.Manifest.Proof.Request != request || r.Manifest.Proof.Evidence.Cluster != s.cluster || r.Commit.Generation == 0 || len(r.Commit.Digest) != 64 {
		return fmt.Errorf("invalid committed job identity")
	}
	switch j.State {
	case JobPrepared, JobStarting, JobRunning, JobExited, JobFailed, JobCanceled, JobExpired:
	default:
		return fmt.Errorf("invalid node job state")
	}
	if r.Workload != nil && (!identifier(r.Workload.Principal) || r.Workload.PreparedVersion == 0 || r.Workload.PreparedVersion >= j.Version) {
		return fmt.Errorf("invalid workload credential intent")
	}
	return validateExecution(r)
}
func (s *Store) saveJob(old []byte, r jobRecord) error {
	if err := s.validateJob(r); err != nil {
		return err
	}
	file, err := jobFile(r.Job.ID)
	if err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > 1<<20 {
		return fmt.Errorf("node job too large")
	}
	return s.path.CompareAndSwap(file, old, b)
}

// PrepareJob publishes signed admission evidence before binding its lease.
// Retrying after a crash between these writes completes the same binding. No
// worker approval can exist until both aggregates agree; missing published job
// history is an error, never an invitation to recreate it.
func (s *Store) PrepareJob(ctx context.Context, actor trust.AuthenticatedPrincipal, accepted nodejob.Accepted, expectedLeaseVersion uint64, now time.Time) (out Job, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	signed, commit, err := accepted.Value()
	if err != nil {
		return out, err
	}
	m := signed.Manifest
	if m.Controller != actor.Principal || m.Node != s.node || now.Unix() < m.Created || now.Unix() >= m.Expires {
		return out, fmt.Errorf("job acceptance actor/target/deadline mismatch")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, leases, err := s.load()
		if err != nil {
			return err
		}
		var l *Lease
		for i := range leases.Leases {
			if leases.Leases[i].ID == m.Lease {
				l = &leases.Leases[i]
				break
			}
		}
		if l == nil || l.Controller != actor.Principal || l.Run != m.Membership.RunID || leases.Active != l.ID || now.Unix() >= l.Expires {
			return fmt.Errorf("job has no live owned reservation")
		}
		if l.State != Reserved && l.State != Prepared {
			return fmt.Errorf("lease no longer preparable")
		}
		b, r, err := s.loadJob(m.Job)
		switch {
		case errors.Is(err, os.ErrNotExist):
			if l.State != Reserved || l.Job != "" {
				return fmt.Errorf("published prepared job missing; reconciliation required")
			}
			request, _ := m.SigningRequest()
			r = jobRecord{Job: Job{Format: JobVersion, ID: m.Job, Attempt: m.Attempt, Lease: m.Lease, Controller: m.Controller, ManifestHash: request.Digest, Version: 1, State: JobPrepared}, Manifest: signed, Commit: commit}
		case err != nil:
			return err
		case digest(r.Manifest) != digest(signed) || r.Commit.Digest != commit.Digest || commit.Generation < r.Commit.Generation:
			return fmt.Errorf("job ID reused with different admission")
		}
		if r.Job.CancelRequested || r.Job.State != JobPrepared {
			return fmt.Errorf("prepared job was canceled or started")
		}
		if l.State == Prepared {
			if l.Job != m.Job {
				return fmt.Errorf("lease bound to another job")
			}
			out = r.Job
			return nil
		}
		if l.Version != expectedLeaseVersion {
			return fmt.Errorf("reservation version changed")
		}
		if b == nil {
			if err := s.saveJob(nil, r); err != nil {
				return err
			}
		}
		l.Job = m.Job
		l.State = Prepared
		l.Version++
		leases.NodeVersion++
		if err := s.save(old, leases); err != nil {
			return err
		}
		out = r.Job
		return nil
	})
	return out, err
}

func (s *Store) JobStatus(actor trust.AuthenticatedPrincipal, id string, now time.Time) (Job, error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return Job{}, err
	}
	_, r, err := s.loadJob(id)
	if err != nil {
		return Job{}, err
	}
	if r.Job.Controller != actor.Principal {
		return Job{}, fmt.Errorf("job belongs to another controller")
	}
	return r.Job, nil
}

// CancelJob records intent, not termination. Hosting must still confirm there
// is no child before any terminal status or accelerator release is published.
func (s *Store) CancelJob(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, expectedVersion uint64, now time.Time) (out Job, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		if r.Job.Controller != actor.Principal {
			return fmt.Errorf("job belongs to another controller")
		}
		if r.Job.CancelRequested {
			out = r.Job
			return nil
		}
		if r.Job.Version != expectedVersion {
			return fmt.Errorf("job version changed")
		}
		switch r.Job.State {
		case JobPrepared, JobStarting, JobRunning:
		default:
			return fmt.Errorf("job is already terminal")
		}
		r.Job.CancelRequested = true
		r.Job.Version++
		if err := s.saveJob(old, r); err != nil {
			return err
		}
		out = r.Job
		return nil
	})
	return out, err
}
