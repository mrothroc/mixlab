package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/admission"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

type DiLoCoCoordinatorOptions struct {
	Principal, AdmissionPrincipal            statehome.Path
	State, Admission, Checkpoints, Artifacts statehome.Path
	Endpoint                                 string
	Clock                                    func() time.Time
	Event                                    func(error)
	Ready                                    func()
}

func ServeInstalledDiLoCo(ctx context.Context, l net.Listener, o DiLoCoCoordinatorOptions) (result error) {
	if o.Clock == nil || o.Ready == nil {
		return fmt.Errorf("coordinator clock and readiness required")
	}
	p, err := principal.Open(o.Principal, o.Clock())
	if err != nil {
		return err
	}
	defer func() { result = errors.Join(result, p.Close()) }()
	a, err := principal.Open(o.AdmissionPrincipal, o.Clock())
	if err != nil {
		return err
	}
	defer func() { result = errors.Join(result, a.Close()) }()
	if err := refreshServicePrincipalStartup(ctx, p, trust.Coordinator, o.Clock); err != nil {
		return err
	}
	if err := refreshServicePrincipalStartup(ctx, a, trust.Controller, o.Clock); err != nil {
		return err
	}
	identity, _, err := p.Active(o.Clock())
	if err != nil {
		return err
	}
	controller, _, err := a.Active(o.Clock())
	if err != nil {
		return err
	}
	if identity.Role != trust.Coordinator || controller.Role != trust.Controller || identity.Cluster != controller.Cluster || identity.Fingerprint != controller.Fingerprint || identity.Principal == controller.Principal {
		return fmt.Errorf("separate enrolled coordinator/controller identities in the same cluster required")
	}
	anchor, err := trust.PinRoot(identity.Root, identity.Fingerprint, o.Clock())
	if err != nil {
		return err
	}
	owner := run.Coordinator{ClusterID: identity.Cluster, PrincipalID: identity.Principal, Role: "coordinator", Audience: "learner-run"}
	if err := validateCoordinatorRoots(o, owner, controller.Principal, false); err != nil {
		return err
	}
	inbox, err := admission.OpenInbox(o.Admission)
	if err != nil {
		return err
	}
	repository, err := coordinatorlocal.Open(o.State)
	if err != nil {
		return err
	}
	store, err := artifactlocal.Open(o.Artifacts)
	if err != nil {
		return err
	}
	artifacts := coordinatorlocal.Artifacts{Store: store}
	checkpoints, err := coordinatorlocal.OpenCheckpoints(o.Checkpoints)
	if err != nil {
		return err
	}
	grants, err := learnerapi.OpenGrants(o.State)
	if err != nil {
		return err
	}
	ledger, err := artifactapi.OpenLedger(o.State)
	if err != nil {
		return err
	}
	models := &DiLoCoModels{Inbox: inbox, Principal: p, Repository: repository, Artifacts: artifacts, Clock: o.Clock}
	sign := func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		if err := ctx.Err(); err != nil {
			return trust.SignedProof{}, err
		}
		s, key, err := p.Active(o.Clock())
		if err != nil {
			return trust.SignedProof{}, err
		}
		v, err := trust.VerifySnapshot(anchor, s.Snapshot, o.Clock())
		if err != nil {
			return trust.SignedProof{}, err
		}
		return trust.SignPrincipalProof(anchor, s.Chain, key, v, q, o.Clock())
	}
	current := learnerTrust(func(ctx context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
		if err := ctx.Err(); err != nil {
			return trust.VerifiedSnapshot{}, err
		}
		s, _, err := p.Active(now)
		if err != nil {
			return trust.VerifiedSnapshot{}, err
		}
		return trust.VerifySnapshot(anchor, s.Snapshot, now)
	})
	models.Recover = func(ctx context.Context, q admission.Submission, m coordinator.Model) (bool, error) {
		nodes := serviceNodes{q, p, o.Clock}
		recovery := DiLoCoRecovery{Model: m, Anchor: anchor, Clock: o.Clock, Sign: sign,
			Current: func(ctx context.Context) (trust.VerifiedSnapshot, error) { return current.Current(ctx, o.Clock()) },
			Node:    func(a run.Allocation) (LearnerRecoveryNode, error) { return nodes.client(a) },
			Grant: func(ctx context.Context, request workload.SignedRequest) (workload.Grant, error) {
				scope := request.Request.Scope
				deadline, err := (nodejob.Manifest{Created: q.Specification.CreatedAt, Expires: q.Deadline, Limits: q.Launch.Selection.Requirements.Limits}).WorkloadDeadline()
				if err != nil {
					return workload.Grant{}, err
				}
				if scope.Deadline != deadline.Unix() {
					return workload.Grant{}, fmt.Errorf("recovery cannot extend original workload deadline")
				}
				return inbox.RecoveryWorkloadGrant(ctx, q.Launch.Run, scope.Job, request, func() (workload.Grant, error) {
					req, err := request.GrantRequest()
					if err != nil {
						return workload.Grant{}, err
					}
					proof, err := sign(ctx, req)
					return workload.Grant{Request: request, Proof: proof}, err
				})
			},
			Issue: func(ctx context.Context, grant workload.Grant) (workload.Result, error) {
				identity, _, err := p.Active(o.Clock())
				if err != nil {
					return workload.Result{}, err
				}
				endpoints := identity.Snapshot.Payload.Endpoints.Payload.URLs
				if len(endpoints) == 0 {
					return workload.Result{}, fmt.Errorf("signed authority endpoint missing")
				}
				return IssueRemoteWorkload(ctx, p, endpoints[0], grant, o.Clock)
			},
		}
		return recovery.Tick(ctx, q.Launch.Run)
	}
	service := DiLoCoService{Inbox: inbox, Clock: o.Clock, Interval: time.Second}
	leases := DiLoCoAdmissionLeases{Repository: repository, Clock: o.Clock, Client: func(role trust.Role, candidate recruitment.Candidate) (AdmissionLeaseClient, error) {
		identity := a
		if role == trust.Coordinator {
			identity = p
		} else if role != trust.Controller {
			return nil, fmt.Errorf("unsupported admission lease owner")
		}
		return NewNodeClient(identity, candidate.Endpoint, candidate.Capabilities.Node, o.Clock)
	}}
	service.Maintain = func(ctx context.Context, q admission.Submission, v admission.Snapshot) error {
		err := leases.Maintain(ctx, q, v)
		if errors.Is(err, ErrAdmissionLeaseLost) && v.Committed {
			model, loadErr := models.Model(ctx, q.Launch.Run)
			if loadErr != nil {
				return errors.Join(err, loadErr)
			}
			return errors.Join(err, model.Fail(ctx, q.Launch.Run, "original allocation lost during admission handoff"))
		}
		return err
	}
	service.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
		m, err := models.Model(ctx, q.Launch.Run)
		if err != nil {
			return admission.Ports{}, err
		}
		return DiLoCoAdmissionPorts(DiLoCoAdmissionOptions{Inbox: inbox, Controller: a, Model: m, Clock: o.Clock, SignCommit: func(ctx context.Context, c run.Commit) (authenticated.Commit, error) {
			proof, err := sign(ctx, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: c.Binding.CommitHash, Context: c.Binding.RunID, Audience: owner.Audience})
			return authenticated.Commit{Body: c, Proof: proof}, err
		}}, q)
	}
	service.Run = func(ctx context.Context, q admission.Submission, v admission.Snapshot) error {
		if err := checkpoints.Sync(ctx, repository, q.Launch.Run); err != nil {
			return err
		}
		if err := models.Tick(ctx, q, v); err != nil {
			return err
		}
		return checkpoints.Sync(ctx, repository, q.Launch.Run)
	}
	service.Event = o.runEvent
	policy := func() (*managedtls.Policy, error) {
		return managedPrincipalPolicyRoles(p, o.Clock, trust.Worker, trust.Node, trust.Controller)
	}
	base := coordinator.Model{Repository: repository, Artifacts: artifacts, Allocations: unselectedAllocations{}, Clock: serviceClock{o.Clock}, Evaluator: remoteEvaluation{}}
	handler := func(policy *managedtls.Policy) (http.Handler, error) {
		learner, err := (learnerapi.Gateway{Policy: policy, Anchor: anchor, Trust: current, Coordinator: owner, Model: base, ModelForRun: models.Model, Clock: o.Clock, Sign: sign, Grants: grants, Timeout: 30 * time.Second}).Handler()
		if err != nil {
			return nil, err
		}
		artifact, err := (artifactapi.Gateway{Policy: policy, Anchor: anchor, Trust: current, Coordinator: owner, Repository: repository, Store: store, Ledger: ledger, Clock: o.Clock, TransferTimeout: time.Minute}).Handler()
		if err != nil {
			return nil, err
		}
		submission, err := (DiLoCoSubmissionAPI{Inbox: inbox, Repository: repository, Policy: policy, ControllerID: controller.Principal, CoordinatorID: owner.PrincipalID, Cluster: owner.ClusterID, Endpoint: o.Endpoint, Clock: o.Clock, Verify: func(ctx context.Context, q admission.Submission) error {
			_, err := artifacts.Load(ctx, q.Specification.BaseModel)
			return err
		}}).Handler()
		if err != nil {
			return nil, err
		}
		mux := http.NewServeMux()
		mux.Handle(learnerapi.Route, learner)
		mux.Handle(artifactapi.ReadRoute, artifact)
		mux.Handle(artifactapi.CreateRoute, artifact)
		mux.Handle(DiLoCoSubmissionRoute, submission)
		mux.Handle(DiLoCoSubmissionRoute+"/", submission)
		mux.Handle(DiLoCoInfoRoute, submission)
		mux.Handle(DiLoCoInitialModelRoute, diLoCoInitialHandler(policy, artifacts))
		return mux, nil
	}
	maintenance := func(ctx context.Context) error {
		life, cancel := context.WithCancel(ctx)
		defer cancel()
		done := make(chan error, 2)
		go func() {
			done <- maintainPrincipalTrust(life, o.Principal, p, trust.Coordinator, o.Clock, o.Event)
			cancel()
		}()
		go func() { done <- MaintainControllerTrust(life, o.AdmissionPrincipal, a, o.Clock, o.Event); cancel() }()
		return errors.Join(<-done, <-done)
	}
	return ServeDiLoCo(ctx, l, DiLoCoServerOptions{Service: service, Policy: policy, Handler: handler, Maintenance: maintenance, Ready: o.Ready})
}

func (o DiLoCoCoordinatorOptions) runEvent(id string, err error) {
	if o.Event != nil {
		o.Event(fmt.Errorf("run %s: %w", id, err))
	}
}

// Restart may follow a public-snapshot expiry. Refresh from a pinned source
// before opening application traffic; this cannot renew an expired leaf or
// bypass revocation, and a currently valid cache tolerates authority outages.
func refreshServicePrincipalStartup(ctx context.Context, p *principal.Store, role trust.Role, clock func() time.Time) error {
	s, err := p.View(clock())
	if err != nil {
		return err
	}
	if s.Role != role {
		return fmt.Errorf("%s principal required", role)
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	var failures error
	for _, endpoint := range s.Snapshot.Payload.Endpoints.Payload.URLs {
		err = RefreshRemoteTrust(ctx, p, endpoint, clock)
		if err == nil {
			_, _, err = p.Active(clock())
			return err
		}
		failures = errors.Join(failures, err)
		if ctx.Err() != nil {
			return errors.Join(failures, ctx.Err())
		}
	}
	_, _, err = p.Active(clock())
	if err == nil {
		return nil
	}
	return errors.Join(err, failures)
}

type unselectedAllocations struct{}

func (unselectedAllocations) Ready(context.Context, run.Slot) error {
	return fmt.Errorf("accepted run-specific allocation required")
}
