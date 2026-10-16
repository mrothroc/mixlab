package nodeagent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
)

// LearnerJob is a separate job variant sharing the node's exclusive lease
// aggregate. Hosting receives only the validated local assignment, never keys.
type LearnerJob struct {
	Signed                nodejob.SignedLearner        `json:"signed"`
	Accepted              trust.AcceptedProof          `json:"accepted"`
	Version               uint64                       `json:"version"`
	Workload              *workloadIntent              `json:"workload,omitempty"`
	Assignment            *workerjob.LearnerAssignment `json:"assignment,omitempty"`
	StartKey              string                       `json:"start_key,omitempty"`
	Stopped               bool                         `json:"stopped"`
	Commit                *run.Commit                  `json:"commit,omitempty"`
	CommitProof           trust.AcceptedProof          `json:"commit_proof"`
	Credential            *workload.Result             `json:"credential,omitempty"`
	Recovery              *nodejob.LearnerRecovery     `json:"recovery,omitempty"`
	History               []nodejob.LearnerRecovery    `json:"attempt_history,omitempty"`
	RetryAcknowledgements []trust.AcceptedProof        `json:"retry_acknowledgements,omitempty"`
	RetryWorkload         *workloadIntent              `json:"retry_workload,omitempty"`
}

func learnerFile(id string) (string, error) {
	if !identifier(id) {
		return "", fmt.Errorf("invalid learner job ID")
	}
	return "learner-" + id + ".json", nil
}
func (s *Store) loadLearner(id string) ([]byte, LearnerJob, error) {
	name, err := learnerFile(id)
	if err != nil {
		return nil, LearnerJob{}, err
	}
	b, err := s.path.ReadFileLimit(name, 2<<20)
	if err != nil {
		return nil, LearnerJob{}, err
	}
	var j LearnerJob
	if err = json.Unmarshal(b, &j); err != nil {
		return nil, j, err
	}
	want, err := json.Marshal(j)
	if err != nil || !bytes.Equal(b, want) {
		return nil, j, fmt.Errorf("noncanonical learner journal")
	}
	return b, j, s.validateLearner(j)
}
func (s *Store) validateLearner(j LearnerJob) error {
	m := j.Signed.Manifest
	q, err := m.SigningRequest()
	if err != nil {
		return err
	}
	if j.Version == 0 || j.Accepted.Generation == 0 || j.Accepted.Digest == "" || m.Slot.Allocation.NodeID != s.node || m.Plan.Coordinator.ClusterID != s.cluster || j.Signed.Proof.Request != q || j.Signed.Proof.Evidence.Principal != m.Controller {
		return fmt.Errorf("invalid learner admission evidence")
	}
	if j.Workload != nil && (!identifier(j.Workload.Principal) || j.Workload.PreparedVersion == 0 || j.Workload.PreparedVersion >= j.Version) {
		return fmt.Errorf("invalid learner credential intent")
	}
	if j.Assignment != nil {
		if err := j.Assignment.Validate(); err != nil {
			return err
		}
		expected := m
		expectedPrivate := ""
		if len(j.History) > 0 {
			expected.Attempt = j.History[len(j.History)-1].Attempt.AttemptID
			expectedPrivate = j.History[len(j.History)-1].Attempt.PrivateHash
		}
		if j.Assignment.ExpectedPrivateHash != expectedPrivate {
			return fmt.Errorf("learner private recovery binding changed")
		}
		if err := matchLearnerAssignment(expected, *j.Assignment); err != nil {
			return err
		}
		if !identifier(j.StartKey) || j.Workload == nil || !j.Workload.Initialized {
			return fmt.Errorf("start without credential preparation")
		}
	}
	if j.Commit != nil {
		if err := j.Commit.ValidateFor(m.Plan); err != nil {
			return err
		}
		if j.CommitProof.Generation == 0 {
			return fmt.Errorf("unverified lease handoff")
		}
	}
	if j.Recovery != nil {
		if j.Commit == nil || !j.Stopped || j.Assignment == nil {
			return fmt.Errorf("retry without stopped committed child")
		}
		if err := j.Recovery.Validate(); err != nil {
			return err
		}
		if j.Recovery.Attempt.Binding != j.Commit.Binding || j.Recovery.Attempt.PreviousAttemptID != j.Assignment.AttemptID || j.Recovery.Attempt.Allocation != m.Slot.Allocation || j.Recovery.Coordinator != m.Plan.Coordinator {
			return fmt.Errorf("retry differs from original allocation")
		}
	}
	if j.Credential != nil {
		if j.Workload == nil || !j.Workload.Initialized {
			return fmt.Errorf("credential without durable key intent")
		}
		b := j.Credential.Binding
		q, _ := m.SigningRequest()
		attempt := m.Attempt
		if len(j.History) > 0 {
			attempt = j.History[len(j.History)-1].Attempt.AttemptID
		}
		if b.Version != trust.LearnerBindingVersion || b.Principal != j.Workload.Principal || b.Attempt != attempt || b.PlanHash != m.Plan.SHA256 || b.ManifestHash != q.Digest || b.Worker != m.Slot.Member.WorkerID || b.Run != m.Plan.Cohort.RunID || b.Coordinator != m.Plan.Coordinator.PrincipalID {
			return fmt.Errorf("saved learner credential mismatch")
		}
	}
	previous, generation := m.Attempt, m.WorkloadGeneration
	if len(j.History) > 1024 {
		return fmt.Errorf("learner retry history full")
	}
	if len(j.RetryAcknowledgements) != len(j.History) {
		return fmt.Errorf("retry history lacks coordinator acknowledgement")
	}
	for i, r := range j.History {
		if j.RetryAcknowledgements[i].Generation == 0 || len(j.RetryAcknowledgements[i].Digest) != 64 {
			return fmt.Errorf("unverified retry acknowledgement")
		}
		if j.Commit == nil {
			return fmt.Errorf("retry without commit")
		}
		if err := r.Validate(); err != nil {
			return err
		}
		if err := r.Attempt.ValidateFor(*j.Commit, previous, generation, r.Attempt.PrivateHash); err != nil {
			return err
		}
		previous, generation = r.Attempt.AttemptID, r.Attempt.WorkloadGeneration
	}
	if j.RetryWorkload != nil && (j.Recovery == nil || !identifier(j.RetryWorkload.Principal) || j.RetryWorkload.PreparedVersion == 0 || j.RetryWorkload.PreparedVersion >= j.Version) {
		return fmt.Errorf("invalid retry credential intent")
	}
	return nil
}
func (s *Store) saveLearner(old []byte, j LearnerJob) error {
	if err := s.validateLearner(j); err != nil {
		return err
	}
	name, _ := learnerFile(j.Signed.Manifest.Slot.Allocation.JobID)
	b, err := json.Marshal(j)
	if err != nil {
		return err
	}
	if len(b) > 2<<20 {
		return fmt.Errorf("learner journal too large")
	}
	return s.path.CompareAndSwap(name, old, b)
}
func (s *Store) PrepareLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, a nodejob.AcceptedLearner, expected uint64, now time.Time) (out LearnerJob, err error) {
	if err = authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	signed, proof, err := a.Value()
	if err != nil {
		return out, err
	}
	m := signed.Manifest
	if m.Controller != actor.Principal || m.Slot.Allocation.NodeID != s.node || now.Unix() < m.Plan.CreatedAt || now.Unix() >= m.Plan.ExpiresAt {
		return out, fmt.Errorf("learner admission expired or wrong owner")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, leases, e := s.load()
		if e != nil {
			return e
		}
		l := findLease(&leases, m.Slot.Allocation.LeaseID)
		if l == nil || l.Controller != actor.Principal || l.Run != m.Plan.Cohort.RunID || l.ID != leases.Active || now.Unix() >= l.Expires || (l.State != Reserved && l.State != Prepared) {
			return fmt.Errorf("live original reservation required")
		}
		b, j, e := s.loadLearner(m.Slot.Allocation.JobID)
		switch {
		case errors.Is(e, os.ErrNotExist):
			if l.State != Reserved || l.Job != "" {
				return fmt.Errorf("prepared learner history missing")
			}
			j = LearnerJob{Signed: signed, Accepted: proof, Version: 1}
		case e != nil:
			return e
		case digest(j.Signed) != digest(signed) || j.Accepted.Digest != proof.Digest || proof.Generation < j.Accepted.Generation:
			return fmt.Errorf("learner job ID reused")
		}
		if j.Stopped || j.Assignment != nil {
			return fmt.Errorf("learner already started or fenced")
		}
		if l.State == Prepared {
			if l.Job != m.Slot.Allocation.JobID || l.JobKind != "learner" {
				return fmt.Errorf("lease bound to another job")
			}
			out = j
			return nil
		}
		if l.Version != expected {
			return fmt.Errorf("reservation version changed")
		}
		if b == nil {
			if e = s.saveLearner(nil, j); e != nil {
				return e
			}
		}
		l.Job = m.Slot.Allocation.JobID
		l.JobKind = "learner"
		l.State = Prepared
		l.Expires = min(l.Expires, m.Plan.ExpiresAt)
		l.RenewBy = min(l.RenewBy, l.Expires)
		l.Version++
		leases.NodeVersion++
		if e = s.save(old, leases); e != nil {
			return e
		}
		out = j
		return nil
	})
	return out, err
}
func matchLearnerAssignment(m nodejob.LearnerManifest, a workerjob.LearnerAssignment) error {
	if a.Cohort == nil || digest(a.Cohort) != digest(m.Plan.Cohort) {
		return fmt.Errorf("local learner cohort differs from admission")
	}
	if a.JobID != m.Slot.Allocation.JobID || a.AttemptID != m.Attempt || a.RunID != m.Plan.Cohort.RunID || a.PlanHash != m.Plan.SHA256 || a.Member != m.Slot.Member || a.Contract != m.Plan.Contract || a.Coordinator != m.Plan.Coordinator || digest(a.Config) != digest(m.Config) || a.DatasetSelector != m.DatasetSelector || a.RuntimeSeconds != m.Limits.RuntimeSeconds || a.PrivateMaxBytes > m.Limits.DiskBytes-m.Limits.LogBytes {
		return fmt.Errorf("local learner assignment differs from admission")
	}
	return nil
}

// StartLearner publishes the attempt binding before exposing it to hosting.
// Resolve the path from the agent-owned dataset catalog, never caller input.
func (s *Store) StartLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, id, key string, now time.Time) (out workerjob.LearnerAssignment, err error) {
	if err = authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(key) {
		return out, fmt.Errorf("start key required")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, e := s.loadLearner(id)
		if e != nil {
			return e
		}
		m := j.Signed.Manifest
		_, leases, e := s.load()
		if e != nil {
			return e
		}
		l := findLease(&leases, m.Slot.Allocation.LeaseID)
		if m.Controller != actor.Principal || j.Stopped || j.Workload == nil || !j.Workload.Initialized || j.Credential == nil || l == nil || l.ID != leases.Active || l.State != Prepared || l.Job != id || now.Unix() >= l.Expires || now.Unix() >= m.Plan.ExpiresAt {
			return fmt.Errorf("learner no longer startable")
		}
		if j.Assignment != nil {
			if j.StartKey != key {
				return fmt.Errorf("start key changed")
			}
			out = *j.Assignment
			return nil
		}
		_, profile, e := s.loadProfile()
		if e != nil {
			return e
		}
		path := ""
		for _, d := range profile.Datasets {
			if d.Selector == m.DatasetSelector && d.ID == m.Plan.Contract.DatasetHash {
				path = d.TrainPattern
			}
		}
		if path == "" || !m.Limits.Within(profile.Limits) || profile.Probe.BuildID != m.Plan.Contract.BuildHash {
			return fmt.Errorf("learner differs from local catalog")
		}
		out = workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: id, AttemptID: m.Attempt, RunID: m.Plan.Cohort.RunID, PlanHash: m.Plan.SHA256, Member: m.Slot.Member, Contract: m.Plan.Contract, Coordinator: m.Plan.Coordinator, Config: m.Config, DatasetSelector: m.DatasetSelector, TrainPattern: path, RuntimeSeconds: m.Limits.RuntimeSeconds, PrivateMaxBytes: min(uint64(8<<30), m.Limits.DiskBytes-m.Limits.LogBytes)}
		out.Cohort = &m.Plan.Cohort
		if e = out.Validate(); e != nil {
			return e
		}
		j.Assignment = &out
		j.StartKey = key
		j.Version++
		return s.saveLearner(old, j)
	})
	return out, err
}

// ConfirmLearnerStopped is LOCAL hosting evidence, not an HTTP operation. The
// full assignment digest prevents an old process outcome from fencing a retry.
func (s *Store) ConfirmLearnerStopped(ctx context.Context, b workercontrol.Binding) error {
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, e := s.loadLearner(b.JobID)
		if e != nil {
			return e
		}
		if j.Assignment == nil {
			return fmt.Errorf("no authorized child")
		}
		want, e := j.Assignment.Binding()
		if e != nil {
			return e
		}
		if want != b {
			return fmt.Errorf("stale child outcome")
		}
		if j.Stopped {
			return nil
		}
		j.Stopped = true
		j.Version++
		return s.saveLearner(old, j)
	})
}

func (s *Store) PrepareLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts) (out workload.SignedRequest, err error) {
	return s.prepareLearnerWorkload(ctx, actor, id, p, false)
}
func (s *Store) PrepareLearnerRetryWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts) (workload.SignedRequest, error) {
	return s.prepareLearnerWorkload(ctx, actor, id, p, true)
}
func (s *Store) prepareLearnerWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, p WorkloadPorts, retry bool) (out workload.SignedRequest, err error) {
	if p.Clock == nil || p.Initialize == nil || p.Request == nil {
		return out, fmt.Errorf("workload ports required")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, j, e := s.loadLearner(id)
		if e != nil {
			return e
		}
		m := j.Signed.Manifest
		intent := func() **workloadIntent {
			if retry {
				return &j.RetryWorkload
			}
			return &j.Workload
		}
		guard := func() error {
			now := p.Clock()
			if e := authorizeLeaseActor(actor, s.cluster, now); e != nil {
				return e
			}
			_, ls, e := s.load()
			if e != nil {
				return e
			}
			l := findLease(&ls, m.Slot.Allocation.LeaseID)
			if l == nil || l.Job != id || ls.Active != l.ID || now.Unix() >= l.Expires {
				return fmt.Errorf("learner workload admission no longer live")
			}
			if retry {
				if actor.Role != trust.Coordinator || !ownsLease(actor, *l) || l.State != Running || !j.Stopped || j.Recovery == nil || now.Unix() >= j.Recovery.Expires {
					return fmt.Errorf("retry workload is not authorized")
				}
			} else if actor.Role != trust.Controller || m.Controller != actor.Principal || j.Stopped || j.Assignment != nil || l.State != Prepared || now.Unix() >= m.Plan.ExpiresAt {
				return fmt.Errorf("initial learner workload is not authorized")
			}
			return ctx.Err()
		}
		if e = guard(); e != nil {
			return e
		}
		if *intent() == nil {
			principal, e := newID()
			if e != nil {
				return e
			}
			*intent() = &workloadIntent{Principal: principal, PreparedVersion: j.Version}
			j.Version++
			if e = s.saveLearner(old, j); e != nil {
				return e
			}
			old, j, e = s.loadLearner(id)
			if e != nil {
				return e
			}
		}
		scope, e := learnerScope(j, (*intent()).Principal, retry)
		if e != nil {
			return e
		}
		if !(*intent()).Initialized {
			if e = p.Initialize(ctx, scope); e != nil {
				return e
			}
			if e = guard(); e != nil {
				return e
			}
			(*intent()).Initialized = true
			j.Version++
			if e = s.saveLearner(old, j); e != nil {
				return e
			}
		}
		out, e = p.Request(ctx, scope)
		if e != nil {
			return e
		}
		if e = guard(); e != nil {
			return e
		}
		if out.Request.Scope != scope {
			return fmt.Errorf("workload port changed learner scope")
		}
		_, e = out.GrantRequest()
		return e
	})
	return out, err
}
