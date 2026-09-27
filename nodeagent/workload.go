package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

// Node management owns permission to create a workload identity. Key material
// and its publication remain in the credential context behind these local ports.
type workloadIntent struct {
	Principal       string `json:"principal"`
	PreparedVersion uint64 `json:"prepared_version"`
	Initialized     bool   `json:"initialized"`
}

type WorkloadPorts struct {
	// Initialize publishes only an empty credential context. It must never
	// generate a key, and exact retry must not replace existing state.
	Initialize func(context.Context, workload.Scope) error
	// Request opens the published context; it must not initialize missing state.
	Request func(context.Context, workload.Scope) (workload.SignedRequest, error)
	Clock   func() time.Time
}

type PreparedWorkload struct {
	Job     Job                    `json:"job"`
	Request workload.SignedRequest `json:"request"`
}

// PrepareWorkload serializes local credential preparation with cancel/cleanup.
// Both ports are local, bounded operations, never remote authority calls. No key
// creation is permitted until the job's initialized marker is durable: a missing
// credential directory after that marker is a fault, not a new identity.
func (s *Store) PrepareWorkload(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, expectedVersion uint64, ports WorkloadPorts) (out PreparedWorkload, err error) {
	if ports.Initialize == nil || ports.Request == nil || ports.Clock == nil || expectedVersion == 0 {
		return out, fmt.Errorf("bounded workload preparation ports and job version required")
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		guard := func() error {
			now := ports.Clock()
			if err := authorize(actor, s.cluster, now); err != nil {
				return err
			}
			_, leases, err := s.load()
			if err != nil {
				return err
			}
			l := findLease(&leases, r.Job.Lease)
			if r.Job.Controller != actor.Principal || r.Job.CancelRequested || r.Job.State != JobPrepared || r.Approval != nil || r.Outcome != nil || l == nil || l.State != Prepared || leases.Active != l.ID || l.Job != id || l.Controller != actor.Principal || now.Unix() >= l.Expires || now.Unix() >= r.Manifest.Manifest.Expires {
				return fmt.Errorf("workload requires a live uncanceled prepared job and lease")
			}
			return ctx.Err()
		}
		if err := guard(); err != nil {
			return err
		}
		if r.Workload != nil && expectedVersion != r.Workload.PreparedVersion && expectedVersion != r.Job.Version {
			return fmt.Errorf("workload retry changed prepared job version")
		}
		if r.Workload == nil {
			if r.Job.Version != expectedVersion || r.Transport != nil {
				return fmt.Errorf("prepared job version changed")
			}
			principal, err := newID()
			if err != nil {
				return err
			}
			r.Workload = &workloadIntent{Principal: principal, PreparedVersion: r.Job.Version}
			r.Job.Version++
			if err := s.saveJob(old, r); err != nil {
				return err
			}
			old, r, err = s.loadJob(id)
			if err != nil {
				return err
			}
		}
		scope, err := r.Manifest.Manifest.WorkloadScope(s.cluster, r.Workload.Principal)
		if err != nil {
			return err
		}
		if !r.Workload.Initialized {
			if err := ports.Initialize(ctx, scope); err != nil {
				return err
			}
			if err := guard(); err != nil {
				return err
			}
			r.Workload.Initialized = true
			r.Job.Version++
			if err := s.saveJob(old, r); err != nil {
				return err
			}
			_, r, err = s.loadJob(id)
			if err != nil {
				return err
			}
		}
		request, err := ports.Request(ctx, scope)
		if err != nil {
			return err
		}
		if err := guard(); err != nil {
			return err
		}
		if request.Request.Scope != scope {
			return fmt.Errorf("credential port returned another workload scope")
		}
		if _, err := request.GrantRequest(); err != nil {
			return err
		}
		out = PreparedWorkload{Job: r.Job, Request: request}
		return nil
	})
	if err != nil {
		return PreparedWorkload{}, err
	}
	return out, nil
}
