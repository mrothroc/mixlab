package nodeagent

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// LocalExecution is an owning-agent view, not a remote response. In particular,
// Approval contains resolved local paths and must never be serialized to HTTP.
type LocalExecution struct {
	Lease     Lease
	Job       *Job
	Manifest  nodejob.Manifest
	Approval  *contract.Approved
	Outcome   *contract.Outcome
	Workload  workload.Scope
	Transport *grouptransport.Signed
}

func (s *Store) ActiveExecution(ctx context.Context) (out *LocalExecution, err error) {
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, leases, err := s.load()
		if err != nil {
			return err
		}
		if leases.Active == "" {
			return nil
		}
		l := findLease(&leases, leases.Active)
		if l == nil {
			return fmt.Errorf("active lease missing")
		}
		out = &LocalExecution{Lease: *l}
		if l.Job == "" {
			return nil
		}
		_, r, err := s.loadJob(l.Job)
		if err != nil {
			return err
		}
		if r.Job.Lease != l.ID || r.Job.Controller != l.Controller {
			return fmt.Errorf("active execution ownership mismatch")
		}
		out.Job, out.Manifest, out.Approval, out.Outcome = &r.Job, r.Manifest.Manifest, r.Approval, r.Outcome
		if r.Workload != nil && r.Workload.Initialized {
			out.Workload, err = r.Manifest.Manifest.WorkloadScope(s.cluster, r.Workload.Principal)
			if err != nil {
				return err
			}
		}
		if r.Transport != nil {
			out.Transport = &r.Transport.Signed
		}
		return nil
	})
	return out, err
}

// ResolveAssignment uses only the administrator's catalog and the committed
// transport's loopback translation. A controller cannot choose a local path.
func (s *Store) ResolveAssignment(ctx context.Context, id string) (out workerjob.Assignment, err error) {
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.loadJob(id)
		if err != nil {
			return err
		}
		if r.Transport == nil {
			return fmt.Errorf("committed transport required")
		}
		_, p, err := s.loadProfile()
		if err != nil {
			return err
		}
		m := r.Manifest.Manifest
		path := ""
		for _, d := range p.Datasets {
			if d.Selector == m.DatasetSelector && d.ID == m.DatasetID {
				path = d.TrainPattern
				break
			}
		}
		if path == "" {
			return fmt.Errorf("prepared dataset registration missing")
		}
		view, err := distributed.NewLocalGroupView(m.Membership, m.Members[m.Rank].MemberID, m.Rank, m.Attempt)
		if err != nil {
			return err
		}
		addresses := make([][]string, len(r.Transport.Signed.Plan.LoopbackPorts))
		for i, port := range r.Transport.Signed.Plan.LoopbackPorts {
			addresses[i] = []string{fmt.Sprintf("127.0.0.1:%d", port)}
		}
		out = workerjob.Assignment{Version: workerjob.Version, JobID: m.Job, AttemptID: m.Attempt, BuildID: m.BuildID, View: view, Config: m.Config, DatasetSelector: m.DatasetSelector, TrainPattern: path, DatasetSHA256: m.DatasetID, ProgramSHA256: m.ProgramHash, RuntimeSeconds: m.Limits.RuntimeSeconds, RingAddresses: addresses}
		out.WeightLayoutSHA256, out.OptimizerSHA256 = m.WeightLayoutHash, m.OptimizerHash
		var input uint64
		for _, a := range m.Artifacts {
			input += a.Bytes
		}
		out.OutputMaxBytes = workerjob.ManagedOutputBudget(m.Limits.DiskBytes, m.Limits.LogBytes, input, m.CheckpointAt > 0)
		out.CheckpointAt = m.CheckpointAt
		return out.Validate()
	})
	return out, err
}

// InterruptActive is a local fail-closed intent for agent restart, shutdown or
// loss of authority. It does not assert child death or release any resource.
// No HTTP request may choose this operation or its reason.
func (s *Store) InterruptActive(ctx context.Context, reason string) error {
	switch reason {
	case "agent_restart", "agent_shutdown", "authority_unavailable":
	default:
		return fmt.Errorf("invalid local interruption reason")
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		if r.Active == "" {
			return nil
		}
		l := findLease(&r, r.Active)
		if l == nil {
			return fmt.Errorf("active lease missing")
		}
		if l.State == Releasing {
			return nil
		}
		l.State, l.CleanupReason = Releasing, reason
		l.Version++
		r.NodeVersion++
		return s.save(old, r)
	})
}

// OwnExecution serializes physical runtime composition across agent processes,
// without holding the state-machine lock while workers or cleanup are running.
func (s *Store) OwnExecution(ctx context.Context, run func(context.Context) error) error {
	if run == nil {
		return fmt.Errorf("execution owner required")
	}
	return s.path.WithProcessLock(ctx, "node-execution-owner.lock", func() error { return run(ctx) })
}
