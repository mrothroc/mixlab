package recruitment

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Run is foreground ownership of exactly one launch. Reopening an interrupted
// attempt compensates it; it never continues starting or automatically retries
// training. Local persistence precedes every potentially ambiguous remote effect.
func (s *Launch) Run(ctx context.Context, p LaunchPorts) error {
	if err := p.validate(); err != nil {
		return err
	}
	return s.path.WithProcessLock(ctx, "launch-owner.lock", func() (err error) {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Phase == "succeeded" {
			return nil
		}
		if r.Phase == "aborted" {
			return fmt.Errorf("launch already aborted; submit a new attempt")
		}
		defer func() {
			if err != nil {
				cleanup, cancel := context.WithTimeout(context.Background(), time.Minute)
				defer cancel()
				err = errors.Join(err, s.compensate(cleanup, p))
			}
		}()
		if r.Phase != "new" {
			return fmt.Errorf("interrupted launch requires compensation, not automatic restart")
		}
		if err = s.launch(ctx, p, &old, &r); err != nil {
			return err
		}
		if err = s.monitor(ctx, p, &old, &r); err != nil {
			return err
		}
		r.Phase = "succeeded"
		return s.save(&old, r)
	})
}

// Abort can be repeated after lost responses, controller restart or timeouts.
// A successful return proves every touched lease reached confirmed RELEASED.
func (s *Launch) Abort(ctx context.Context, p LaunchPorts) error {
	if p.AbortReservation == nil || p.LeaseStatus == nil {
		return fmt.Errorf("authenticated reservation abort and status ports required")
	}
	return s.path.WithProcessLock(ctx, "launch-owner.lock", func() error { return s.compensate(ctx, p) })
}

func (s *Launch) phase(old *[]byte, r *launchRecord, phase string) error {
	r.Phase = phase
	return s.save(old, *r)
}

func (s *Launch) launch(ctx context.Context, p LaunchPorts, old *[]byte, r *launchRecord) error {
	if err := s.phase(old, r, "reserving"); err != nil {
		return err
	}
	for i := range r.Nodes {
		n := &r.Nodes[i]
		n.Touched = true
		if err := s.save(old, *r); err != nil {
			return err
		}
		lease, err := p.Reserve(ctx, n.Candidate, n.Reserve)
		if err != nil {
			return fmt.Errorf("reserve node %s: %w", n.Candidate.Capabilities.Node, err)
		}
		if !leaseMatches(r.Plan, *n, lease) || lease.State != nodeagent.Reserved || lease.Expires <= p.Clock().Unix() {
			return fmt.Errorf("invalid reservation response")
		}
		n.Lease = &lease
		if err := s.save(old, *r); err != nil {
			return err
		}
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
	}
	leases := make([]nodeagent.Lease, len(r.Nodes))
	for i, n := range r.Nodes {
		leases[i] = *n.Lease
	}
	jobs, err := p.BuildJobs(ctx, r.Plan, leases)
	if err != nil {
		return err
	}
	if len(jobs) != len(r.Nodes) {
		return fmt.Errorf("job builder changed cohort size")
	}
	for i, j := range jobs {
		if err := checkLaunchManifest(r.Plan, r.Nodes[i], i, j); err != nil {
			return err
		}
		if i > 0 && (!reflect.DeepEqual(j.Manifest.Membership, jobs[0].Manifest.Membership) || j.Manifest.ConfigHash != jobs[0].Manifest.ConfigHash || j.Manifest.ProgramHash != jobs[0].Manifest.ProgramHash || j.Manifest.WeightLayoutHash != jobs[0].Manifest.WeightLayoutHash || j.Manifest.OptimizerHash != jobs[0].Manifest.OptimizerHash) {
			return fmt.Errorf("cohort manifests disagree")
		}
		r.Nodes[i].Manifest = &jobs[i]
	}
	if err := s.phase(old, r, "preparing"); err != nil {
		return err
	}
	for i := range r.Nodes {
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
		n := &r.Nodes[i]
		prepared, err := p.Prepare(ctx, n.Candidate, n.Lease.Version, *n.Manifest)
		if err != nil {
			return err
		}
		m := n.Manifest.Manifest
		want, err := m.WorkloadScope(r.Plan.Cluster, prepared.Request.Request.Scope.Workload)
		if err != nil {
			return err
		}
		if prepared.Request.Request.Scope != want || prepared.Job.ID != m.Job || prepared.Job.Lease != m.Lease || prepared.Job.State != nodeagent.JobPrepared || prepared.Job.Attempt != m.Attempt || prepared.Job.ManifestHash != n.Manifest.Proof.Request.Digest || prepared.Job.CancelRequested {
			return fmt.Errorf("prepared workload differs from job")
		}
		n.Prepared = &prepared
		if err := s.save(old, *r); err != nil {
			return err
		}
	}
	if err := s.phase(old, r, "issuing"); err != nil {
		return err
	}
	for i := range r.Nodes {
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
		n := &r.Nodes[i]
		grant, err := p.Grant(ctx, n.Prepared.Request)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(grant.Request, n.Prepared.Request) {
			return fmt.Errorf("grant substituted prepared request")
		}
		n.Grant = &grant
		if err := s.save(old, *r); err != nil {
			return err
		}
		credential, err := p.Issue(ctx, grant)
		if err != nil {
			return err
		}
		n.Credential = &credential
		if err := s.save(old, *r); err != nil {
			return err
		}
	}
	results := make([]workload.Result, len(r.Nodes))
	for i, n := range r.Nodes {
		results[i] = *n.Credential
	}
	transport, err := p.BuildTransport(ctx, r.Plan, jobs, results)
	if err != nil {
		return err
	}
	if err := transport.Plan.Validate(); err != nil {
		return err
	}
	q, err := transport.Plan.SigningRequest()
	if err != nil {
		return err
	}
	if transport.Proof.Request != q || transport.Plan.Cluster != r.Plan.Cluster || transport.Plan.Controller != r.Plan.Controller || transport.Plan.Attempt != r.Plan.Attempt || !reflect.DeepEqual(transport.Plan.Membership, jobs[0].Manifest.Membership) {
		return fmt.Errorf("transport differs from fixed launch")
	}
	if len(transport.Plan.Members) != len(r.Nodes) {
		return fmt.Errorf("transport changed cohort size")
	}
	for i, n := range r.Nodes {
		m := transport.Plan.Members[i]
		if m.Node != n.Candidate.Capabilities.Node || m.Job != n.Prepared.Job.ID || m.ManifestHash != n.Prepared.Job.ManifestHash || m.Endpoint != n.Candidate.Capabilities.TransportEndpoint || !reflect.DeepEqual(m.Chain, n.Credential.Chain) || m.Binding != n.Credential.Binding {
			return fmt.Errorf("transport changed prepared member")
		}
	}
	r.Transport = &transport
	if err := s.phase(old, r, "activating"); err != nil {
		return err
	}
	for i := range r.Nodes {
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
		n := &r.Nodes[i]
		j, err := p.Activate(ctx, n.Candidate, n.Prepared.Job.Version, transport, *n.Grant, *n.Credential)
		if err != nil {
			return err
		}
		if !jobMatches(*n, j) || j.State != nodeagent.JobPrepared || j.TransportHash != q.Digest || j.CancelRequested {
			return fmt.Errorf("transport acknowledgement mismatch")
		}
		n.Activated = &j
		n.Start.Job, n.Start.ExpectedVersion = j.ID, j.Version
		if err := s.save(old, *r); err != nil {
			return err
		}
	}
	if err := s.phase(old, r, "starting"); err != nil {
		return err
	}
	for i := range r.Nodes {
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
		n := &r.Nodes[i]
		j, err := p.Start(ctx, n.Candidate, n.Start)
		if err != nil {
			return err
		}
		if !jobMatches(*n, j) || j.TransportHash != q.Digest || j.CancelRequested || (j.State != nodeagent.JobStarting && j.State != nodeagent.JobRunning && j.State != nodeagent.JobExited) {
			return fmt.Errorf("start acknowledgement mismatch")
		}
		n.Started = &j
		if err := s.save(old, *r); err != nil {
			return err
		}
	}
	return s.phase(old, r, "running")
}

func jobMatches(n launchNode, j nodeagent.Job) bool {
	return n.Manifest != nil && j.Format == nodeagent.JobVersion && j.ID == n.Manifest.Manifest.Job && j.Lease == n.Manifest.Manifest.Lease && j.Attempt == n.Manifest.Manifest.Attempt && j.Controller == n.Manifest.Manifest.Controller && j.ManifestHash == n.Manifest.Proof.Request.Digest && j.Version > 0
}

func (s *Launch) renewLeases(ctx context.Context, p LaunchPorts, old *[]byte, r *launchRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	for i := range r.Nodes {
		n := &r.Nodes[i]
		if n.Lease == nil || n.Released {
			continue
		}
		l, err := p.LeaseStatus(ctx, n.Candidate, n.Lease.ID)
		if err != nil {
			return err
		}
		if !leaseMatches(r.Plan, *n, l) || l.ID != n.Lease.ID || l.Version < n.Lease.Version {
			return fmt.Errorf("lease status identity mismatch")
		}
		if l.State == nodeagent.Releasing || l.State == nodeagent.Released {
			// Only a fully started job can complete successfully while monitored.
			if r.Phase != "running" || n.Started == nil {
				return fmt.Errorf("lease ended during launch")
			}
			continue
		}
		if l.Expires <= p.Clock().Unix() {
			return fmt.Errorf("lease expired during launch")
		}
		if n.Renewal == nil && p.Clock().Unix() < l.RenewBy {
			n.Lease = &l
			continue
		}
		if n.Renewal == nil {
			key, err := randomID()
			if err != nil {
				return err
			}
			n.Renewal = &nodeagent.LeaseCommand{IdempotencyKey: key, Lease: l.ID, ExpectedVersion: l.Version, TTLSeconds: r.Plan.TTLSeconds}
			if err := s.save(old, *r); err != nil {
				return err
			}
		}
		updated, err := p.Renew(ctx, n.Candidate, *n.Renewal)
		if err != nil {
			return err
		}
		if !leaseMatches(r.Plan, *n, updated) || updated.ID != l.ID || updated.Version <= n.Renewal.ExpectedVersion || updated.Expires <= p.Clock().Unix() || (updated.State != nodeagent.Reserved && updated.State != nodeagent.Prepared && updated.State != nodeagent.Running) {
			return fmt.Errorf("renewal identity mismatch")
		}
		n.Renewal = nil
		n.Lease = &updated
		if err := s.save(old, *r); err != nil {
			return err
		}
	}
	return nil
}

func (s *Launch) monitor(ctx context.Context, p LaunchPorts, old *[]byte, r *launchRecord) error {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		if err := s.renewLeases(ctx, p, old, r); err != nil {
			return err
		}
		complete := true
		for i := range r.Nodes {
			n := &r.Nodes[i]
			j, err := p.Status(ctx, n.Candidate, n.Manifest.Manifest.Job)
			if err != nil {
				return err
			}
			if !jobMatches(*n, j) || j.CancelRequested || j.TransportHash != n.Activated.TransportHash || j.Version < n.Started.Version {
				return fmt.Errorf("job ownership changed or cancellation requested")
			}
			switch j.State {
			case nodeagent.JobExited:
			case nodeagent.JobStarting, nodeagent.JobRunning:
				complete = false
			default:
				return fmt.Errorf("managed rank terminated without success: %s", j.State)
			}
			l, err := p.LeaseStatus(ctx, n.Candidate, n.Lease.ID)
			if err != nil {
				return err
			}
			if !leaseMatches(r.Plan, *n, l) || l.ID != n.Lease.ID || l.Version < n.Lease.Version {
				return fmt.Errorf("terminal lease identity mismatch")
			}
			if l.State != nodeagent.Released {
				complete = false
			} else if j.State != nodeagent.JobExited {
				return fmt.Errorf("released lease without successful worker")
			}
		}
		if complete {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}
