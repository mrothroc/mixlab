package clusterapp

import (
	"context"
	"fmt"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

func (a *NodeJobs) PrepareLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, q NodeLearnerPrepareRequest) (out NodeLearnerPrepared, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.options.LearnerProbe == nil {
		return out, fmt.Errorf("learner numerical probe unavailable")
	}
	v, err := a.current(ctx)
	if err != nil {
		return out, err
	}
	accepted, err := nodejob.AcceptLearner(a.options.Anchor, v, actor, a.options.Node, q.Signed, a.options.Clock())
	if err != nil {
		return out, err
	}
	j, err := a.options.Store.PrepareLearnerChecked(ctx, actor, accepted, q.ExpectedLeaseVersion, a.options.Preparation)
	if err != nil {
		return out, err
	}
	m := j.Signed.Manifest
	p, err := a.options.LearnerProbe(ctx, m.Config)
	if err != nil {
		return out, err
	}
	if err := p.Validate(); err != nil {
		return out, err
	}
	c := m.Plan.Contract
	if p.Learner == nil || p.Learner.Precision == nil || p.BuildID != c.BuildHash || p.ConfigHash != c.ConfigHash || p.ProgramHash != c.ProgramHash || p.WeightLayoutHash != c.WeightLayoutHash || p.OptimizerHash != c.InnerOptimizerHash || p.Learner.Outer != m.Plan.Outer {
		return out, fmt.Errorf("approved learner numerical contract changed")
	}
	request, err := m.SigningRequest()
	if err != nil {
		return out, err
	}
	preparation, err := run.Digest(struct {
		Probe    any
		Manifest string
	}{p, request.Digest})
	if err != nil {
		return out, err
	}
	precisionHash, err := p.Learner.Precision.Digest()
	if err != nil {
		return out, err
	}
	out.Registration = run.Registration{Format: run.RegistrationVersion, RunID: m.Plan.Cohort.RunID, PlanHash: m.Plan.SHA256, Generation: m.Plan.Cohort.Generation, CohortHash: m.Plan.Cohort.CohortHash, Slot: m.Slot, Contract: c, BaseModelHash: m.Plan.BaseModel.Ref.SHA256, PreparationHash: preparation, JobManifestHash: request.Digest, AttemptID: m.Attempt, WorkloadGeneration: m.WorkloadGeneration, Coordinator: m.Plan.Coordinator, Precision: *p.Learner.Precision, PrecisionHash: precisionHash}
	if err := out.Registration.ValidateFor(m.Plan); err != nil {
		return out, err
	}
	if _, err = a.current(ctx); err != nil {
		return out, err
	}
	out.Request, err = a.options.Store.PrepareLearnerWorkload(ctx, actor, m.Slot.Allocation.JobID, a.options.Workload)
	return out, err
}

func (a *NodeJobs) CredentialLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q NodeLearnerCredentialRequest) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return err
	}
	// Persist verified public issuance before installing the certificate. A
	// retry can complete installation; cleanup already owns the private key.
	if err := a.options.Store.ActivateLearnerCredential(ctx, actor, id, a.options.Anchor, q.Grant, q.Credential, a.options.Clock()); err != nil {
		return err
	}
	dir, err := workloadDirectory(a.options.CredentialRoot, q.Grant.Request.Request.Scope)
	if err != nil {
		return err
	}
	c, err := nodecredentials.OpenTransport(dir, a.options.Anchor, q.Grant.Request.Request.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = c.Close() }()
	return c.Install(ctx, q.Grant, q.Credential, v, a.options.Clock())
}

func (a *NodeJobs) StartLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q NodeLearnerStartRequest) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return err
	}
	if err = a.options.Store.CheckExecutionTrust(ctx, a.options.Anchor, v, a.options.Clock()); err != nil {
		return err
	}
	_, err = a.options.Store.StartLearner(ctx, actor, id, q.Key, a.options.Clock())
	return err
}

func (a *NodeJobs) FinalizeLearner(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, q authenticated.Commit) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	v, err := a.current(ctx)
	if err != nil {
		return err
	}
	return a.options.Store.FinalizeLearner(ctx, actor, id, a.options.Anchor, v, q, a.options.Clock())
}

var _ NodeLearnerOperations = (*NodeJobs)(nil)
