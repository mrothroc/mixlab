package clusterapp

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

type DiLoCoAdmissionOptions struct {
	Inbox      *admission.Inbox
	Controller *principal.Store
	Model      coordinator.Model
	SignCommit func(context.Context, run.Commit) (authenticated.Commit, error)
	Clock      func() time.Time
}

// DiLoCoWorkerID is stable across lost admission responses and service restarts.
func DiLoCoWorkerID(runID, attempt, node string) string {
	h, _ := run.Digest(struct{ Kind, Run, Attempt, Node string }{"learner", runID, attempt, node})
	return h[:32]
}
func buildDiLoCoPlan(q admission.Submission, leases []nodeagent.Lease) (run.Plan, error) {
	s, l := q.Specification, q.Launch
	p := run.Plan{Format: run.PlanVersion, Coordinator: s.Coordinator, Contract: s.Contract, BaseModel: s.BaseModel, Outer: s.Outer, ArtifactStoreID: s.ArtifactStoreID, Endpoint: s.Endpoint, CreatedAt: s.CreatedAt, ExpiresAt: q.Deadline}
	if len(leases) != len(l.Selection.Selected) {
		return p, fmt.Errorf("reserved cohort changed")
	}
	members := make([]distributed.LearnerMember, len(leases))
	for i, lease := range leases {
		c := l.Selection.Selected[i]
		if lease.Node != c.Capabilities.Node || lease.Controller != l.Controller || lease.Run != l.Run {
			return p, fmt.Errorf("reservation differs from selected node")
		}
		worker := DiLoCoWorkerID(l.Run, l.Attempt, lease.Node)
		job, _ := run.Digest(struct{ Run, Attempt, Node, Kind string }{l.Run, l.Attempt, lease.Node, "job"})
		members[i] = distributed.LearnerMember{WorkerID: worker, ParticipantID: lease.Node, IslandID: "single-metal", DatasetPartitionID: fmt.Sprintf("partition-%d", i)}
		a := run.Allocation{Provider: "node_agent", NodeID: lease.Node, LeaseID: lease.ID, JobID: job[:32]}
		a.RefHash, _ = run.Digest(a)
		h, err := run.Digest(c.Capabilities)
		if err != nil {
			return p, err
		}
		p.Slots = append(p.Slots, run.Slot{Member: members[i], CapabilityHash: h, Allocation: a})
	}
	var err error
	p.Cohort, err = distributed.NewLearnerCohort(l.Run, 1, members)
	if err != nil {
		return p, err
	}
	p.SHA256, err = p.Digest()
	if err != nil {
		return p, err
	}
	return p, p.Validate()
}

func DiLoCoAdmissionPorts(o DiLoCoAdmissionOptions, q admission.Submission) (admission.Ports, error) {
	var out admission.Ports
	if o.Inbox == nil || o.Controller == nil || o.SignCommit == nil || o.Clock == nil {
		return out, fmt.Errorf("durable admission and protected identity ports required")
	}
	if err := q.Validate(); err != nil {
		return out, err
	}
	identity, _, err := o.Controller.Active(o.Clock())
	if err != nil {
		return out, err
	}
	if identity.Role != trust.Controller || identity.Principal != q.Launch.Controller || identity.Cluster != q.Launch.Cluster {
		return out, fmt.Errorf("service admission identity changed")
	}
	client := func(i int) (*NodeClient, error) {
		if i < 0 || i >= len(q.Launch.Selection.Selected) {
			return nil, fmt.Errorf("invalid learner slot")
		}
		c := q.Launch.Selection.Selected[i]
		return NewNodeClient(o.Controller, c.Endpoint, c.Capabilities.Node, o.Clock)
	}
	find := func(p run.Plan, r run.Registration) (int, error) {
		if err := r.ValidateFor(p); err != nil {
			return 0, err
		}
		for i, s := range p.Slots {
			if s == r.Slot {
				return i, nil
			}
		}
		return 0, fmt.Errorf("registration not in plan")
	}
	signer := RecruitmentOptions{Principal: o.Controller, Clock: o.Clock}
	out.Clock = func() int64 { return o.Clock().Unix() }
	out.Reserve = func(ctx context.Context, c recruitment.Candidate, r nodeagent.Reserve) (nodeagent.Lease, error) {
		n, err := NewNodeClient(o.Controller, c.Endpoint, c.Capabilities.Node, o.Clock)
		if err != nil {
			return nodeagent.Lease{}, err
		}
		return n.Reserve(ctx, r)
	}
	out.AbortReservation = func(ctx context.Context, c recruitment.Candidate, r nodeagent.Reserve) error {
		n, err := NewNodeClient(o.Controller, c.Endpoint, c.Capabilities.Node, o.Clock)
		if err != nil {
			return err
		}
		a, err := n.AbortReservation(ctx, r)
		if err != nil {
			return err
		}
		if a.Lease != nil && a.Lease.State != nodeagent.Released {
			return fmt.Errorf("original reservation cleanup pending")
		}
		return nil
	}
	out.BuildPlan = func(_ context.Context, _ recruitment.LaunchPlan, _ admission.Specification, l []nodeagent.Lease) (run.Plan, error) {
		return buildDiLoCoPlan(q, l)
	}
	out.Stage = o.Model.Stage
	out.Prepare = func(ctx context.Context, p run.Plan, _ admission.Specification, i int) (run.Registration, error) {
		var zero run.Registration
		n, err := client(i)
		if err != nil {
			return zero, err
		}
		signed, err := o.Inbox.Manifest(ctx, q.Launch.Run, p.Slots[i].Allocation.JobID, func() (nodejob.SignedLearner, error) {
			m := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: q.Launch.Controller, Attempt: q.Launch.Attempt, WorkloadGeneration: 1, Plan: p, Slot: p.Slots[i], Config: q.Specification.Config, DatasetSelector: q.Launch.Selection.Requirements.DatasetSelector, Limits: q.Launch.Selection.Requirements.Limits}
			policy := q.Specification.Policy
			if policy.WorkerID == m.Slot.Member.WorkerID {
				m.Validation = &run.ValidationDataset{Hash: policy.ValidationHash, Split: policy.Split, BatchTokens: policy.BatchTokens, SequenceLength: policy.SequenceLength, Batches: policy.Batches}
			}
			req, err := m.SigningRequest()
			if err != nil {
				return nodejob.SignedLearner{}, err
			}
			proof, err := signer.sign(ctx, req)
			return nodejob.SignedLearner{Manifest: m, Proof: proof}, err
		})
		if err != nil {
			return zero, err
		}
		lease, err := n.LeaseStatus(ctx, p.Slots[i].Allocation.LeaseID)
		if err != nil {
			return zero, err
		}
		prepared, err := n.PrepareLearner(ctx, NodeLearnerPrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: signed})
		if err != nil {
			return zero, err
		}
		grant, err := o.Inbox.WorkloadGrant(ctx, q.Launch.Run, p.Slots[i].Allocation.JobID, prepared.Request, func() (workload.Grant, error) {
			return signer.grantWithAuthorization(ctx, prepared.Request, prepared.Authorization)
		})
		if err != nil {
			return zero, err
		}
		authorized, err := signer.authorizeGrant(ctx, grant, prepared.Authorization)
		if err != nil {
			return zero, err
		}
		identity, _, err := o.Controller.Active(o.Clock())
		if err != nil {
			return zero, err
		}
		endpoints := identity.Snapshot.Payload.Endpoints.Payload.URLs
		if len(endpoints) == 0 {
			return zero, fmt.Errorf("signed authority endpoint missing")
		}
		credential, err := IssueRemoteAuthorizedWorkload(ctx, o.Controller, endpoints[0], authorized, o.Clock)
		if err != nil {
			return zero, err
		}
		if err := n.CredentialLearner(ctx, p.Slots[i].Allocation.JobID, NodeLearnerCredentialRequest{Grant: grant, Credential: credential}); err != nil {
			return zero, err
		}
		return prepared.Registration, nil
	}
	out.Start = func(ctx context.Context, p run.Plan, r run.Registration) error {
		i, err := find(p, r)
		if err != nil {
			return err
		}
		n, err := client(i)
		if err != nil {
			return err
		}
		h, _ := run.Digest(struct{ Plan, Job, Kind string }{p.SHA256, r.Slot.Allocation.JobID, "start"})
		return n.StartLearner(ctx, r.Slot.Allocation.JobID, NodeLearnerStartRequest{Key: h[:32]})
	}
	out.Register = func(ctx context.Context, r run.Registration) error {
		s, err := o.Model.Repository.Load(ctx, r.RunID)
		if err != nil {
			return err
		}
		return r.ValidateFor(s.Plan)
	}
	out.Commit = func(ctx context.Context, c run.Commit) error { return o.Model.Commit(ctx, c.Binding.RunID, c) }
	out.Abort = o.Model.FenceAdmission
	out.Finalize = func(ctx context.Context, p run.Plan, c run.Commit, i int) error {
		n, err := client(i)
		if err != nil {
			return err
		}
		signed, err := o.SignCommit(ctx, c)
		if err != nil {
			return err
		}
		return n.FinalizeLearner(ctx, p.Slots[i].Allocation.JobID, signed)
	}
	return out, nil
}
