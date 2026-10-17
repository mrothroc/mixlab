package integration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/coordinator/nodeleases"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

type recoveryClock struct{ x *learnerHTTPFixture }

func (c recoveryClock) Unix() int64 { return c.x.f.now.Unix() }

type recoveryLeases struct{ node *clusterapp.NodeClient }

func (n recoveryLeases) Status(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
	return n.node.LeaseStatus(ctx, a.LeaseID)
}
func (n recoveryLeases) Renew(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return n.node.Renew(ctx, q)
}
func (n recoveryLeases) Release(ctx context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return n.node.Release(ctx, q)
}

type recoveryEvaluator struct {
	repo  coordinator.Repository
	clock coordinator.Clock
}

func (e recoveryEvaluator) Evaluate(ctx context.Context, a run.EvaluationAssignment) (run.EvaluationResult, error) {
	s, err := e.repo.Load(ctx, a.Binding.RunID)
	if err != nil {
		return run.EvaluationResult{}, err
	}
	h, _ := run.Digest(a)
	return run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: a, AssignmentHash: h, Loss: 1, EffectiveTokens: 8, StartedAt: e.clock.Unix(), FinishedAt: e.clock.Unix(), PrivateBeforeHash: s.PrivateHashes[0], PrivateAfterHash: s.PrivateHashes[0]}, nil
}

type recoveryFixture struct {
	x            *learnerHTTPFixture
	model        coordinator.Model
	repoPath     statehome.Path
	inboxPath    statehome.Path
	node         *clusterapp.NodeClient
	issuer       *workload.Service
	assignment   run.Assignment
	update       private.Update
	fault        string
	issuedGrants []string
}

func newRecoveryFixture(t *testing.T) *recoveryFixture {
	return newRecoveryFixtureWithBuild(t, strings.Repeat("a", 64), true)
}

func newRecoveryFixtureWithBuild(t *testing.T, build string, recordStart bool) *recoveryFixture {
	t.Helper()
	ctx := context.Background()
	blobs, err := artifactlocal.Open(learnerHTTPPath(t, statehome.Principal))
	check(t, err)
	artifacts := coordinatorlocal.Artifacts{Store: blobs}
	base, err := artifacts.Put(ctx, "model_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{4}}})
	check(t, err)
	x := newLearnerHTTPWithBuild(t, func(p *run.Plan) { p.BaseModel = base; p.Outer.TotalSteps = p.Outer.InnerSteps }, build)
	controller, _ := x.client(t, x.f.client)
	node, _ := x.client(t, x.coordinator)
	prepared, err := controller.PrepareLearner(ctx, x.prepare)
	check(t, err)
	req, err := prepared.Request.GrantRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, req, x.f.now)
	check(t, err)
	grant := workload.Grant{Request: prepared.Request, Proof: proof}
	issuer, err := workload.Initialize(ctx, learnerHTTPPath(t, statehome.Authority), workload.Authority{Anchor: x.f.a, Issuer: x.f.issuer, Key: x.f.issuerKey}, x.f.view, x.f.now)
	check(t, err)
	result, err := issuer.Issue(ctx, x.f.client.Chain, grant, x.f.view, x.f.now)
	check(t, err)
	job := x.prepare.Signed.Manifest.Slot.Allocation.JobID
	check(t, controller.CredentialLearner(ctx, job, clusterapp.NodeLearnerCredentialRequest{Grant: grant, Credential: result}))
	check(t, controller.StartLearner(ctx, job, clusterapp.NodeLearnerStartRequest{Key: id(t)}))
	p := x.prepare.Signed.Manifest.Plan
	commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: p.Cohort.RunID, PlanHash: p.SHA256, Generation: p.Cohort.Generation, CohortHash: p.Cohort.CohortHash}, Coordinator: p.Coordinator, Registrations: []run.Registration{prepared.Registration}, CommittedAt: x.f.now.Unix()}
	commit.Binding.CommitHash, err = commit.Digest()
	check(t, err)
	proof, err = trust.SignPrincipalProof(x.f.a, x.coordinator.Chain, x.coordinator.Key, x.f.view, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: commit.Binding.CommitHash, Context: p.Cohort.RunID, Audience: p.Coordinator.Audience}, x.f.now)
	check(t, err)
	check(t, node.FinalizeLearner(ctx, job, authenticated.Commit{Body: commit, Proof: proof}))
	f := &recoveryFixture{x: x, node: node, issuer: issuer, repoPath: learnerHTTPPath(t, statehome.Principal)}
	f.inboxPath = learnerHTTPPath(t, statehome.Principal)
	repo, err := coordinatorlocal.Open(f.repoPath)
	check(t, err)
	clock := recoveryClock{x}
	f.model = coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: recoveryEvaluator{repo, clock}}
	keeper := &nodeleases.Keeper{Repository: repo, Nodes: recoveryLeases{node}, Clock: clock, RunID: p.Cohort.RunID, TTLSeconds: 300, Fail: f.model.Fail}
	f.model.Allocations = keeper
	policy := coordinator.EvaluationPolicy{WorkerID: p.Slots[0].Member.WorkerID, ValidationHash: p.Contract.DatasetHash, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1, TimeoutSeconds: 200}
	check(t, f.model.Stage(ctx, p, policy))
	inbox, err := admission.OpenInbox(f.inboxPath)
	check(t, err)
	requirements := x.requirements
	requirements.Count = 2
	m := x.prepare.Signed.Manifest
	// Admission requires at least two selected candidates. Only this worker's
	// already-committed recovery is exercised; no admission effects run here.
	other := x.capabilities
	other.Node = id(t)
	check(t, inbox.Accept(ctx, admission.Submission{Owner: m.Controller, Deadline: p.ExpiresAt, Launch: recruitment.LaunchPlan{Cluster: p.Coordinator.ClusterID, Controller: m.Controller, Run: p.Cohort.RunID, Attempt: m.Attempt, TTLSeconds: 300, Selection: recruitment.Selection{Policy: recruitment.SelectionPolicy, Requirements: requirements, Selected: []recruitment.Candidate{{Endpoint: strings.TrimPrefix(x.url, "https://"), Capabilities: x.capabilities}, {Endpoint: "unused.local:7445", Capabilities: other}}}}, Specification: admission.Specification{Contract: p.Contract, BaseModel: p.BaseModel, Outer: p.Outer, Coordinator: p.Coordinator, ArtifactStoreID: p.ArtifactStoreID, Endpoint: p.Endpoint, Config: m.Config, CreatedAt: p.CreatedAt, Policy: policy}}))
	check(t, f.model.Commit(ctx, p.Cohort.RunID, commit))
	assignments, err := f.model.OpenRound(ctx, p.Cohort.RunID, 200)
	check(t, err)
	f.assignment = assignments[0]
	ah, _ := run.Digest(f.assignment)
	if recordStart {
		check(t, f.model.RecordStart(ctx, private.Start{Format: "mixlab_learner_start_v1", Binding: commit.Binding, WorkerID: f.assignment.WorkerID, AssignmentHash: ah, PrivateHash: p.Contract.ConfigHash}))
	}
	delta, err := artifacts.Put(ctx, "delta_f32", []optimizer.Tensor{{Name: "w0", Shape: []int{1}, Values: []float32{1}}})
	check(t, err)
	a := f.assignment
	f.update = private.Update{Format: "mixlab_outer_update_v1", ID: "recovered-update", AssignmentHash: ah, Binding: a.Binding, Round: a.Round, BaseRound: a.BaseRound, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: p.Slots[0].Member.ParticipantID, IslandID: p.Slots[0].Member.IslandID, Contract: a.Contract, Delta: delta, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: p.Contract.DatasetHash, Attempted: a.InnerSteps, Committed: a.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 2, Loss: 2, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: x.f.now.Unix(), FinishedAt: x.f.now.Unix()}
	return f
}

func (f *recoveryFixture) state(t *testing.T) coordinator.State {
	t.Helper()
	s, e := f.model.Repository.Load(context.Background(), f.assignment.Binding.RunID)
	check(t, e)
	return s
}
func (f *recoveryFixture) stop(t *testing.T) {
	t.Helper()
	x, e := f.x.store.ActiveExecution(context.Background())
	check(t, e)
	hash, e := run.Digest(*x.Approval)
	check(t, e)
	check(t, f.x.store.ApplyLearnerOutcome(context.Background(), contract.Outcome{Version: 1, JobID: x.Job.ID, AttemptID: x.Approval.AttemptID(), ManifestHash: x.Approval.ManifestHash, ApprovalHash: hash, Kind: contract.Failed, NoChild: true, Error: "injected child exit"}))
}

type recoveryLostNode struct {
	clusterapp.LearnerRecoveryNode
	f *recoveryFixture
}

func (n recoveryLostNode) AuthorizeLearnerRetry(ctx context.Context, id string, q clusterapp.NodeLearnerRetryRequest) (clusterapp.NodeLearnerRetryPrepared, error) {
	out, err := n.LearnerRecoveryNode.AuthorizeLearnerRetry(ctx, id, q)
	if err == nil && n.f.fault == "authorize" {
		n.f.fault = ""
		err = io.ErrUnexpectedEOF
	}
	return out, err
}
func (n recoveryLostNode) StartLearnerRetry(ctx context.Context, id string, q clusterapp.NodeLearnerRetryStart) error {
	err := n.LearnerRecoveryNode.StartLearnerRetry(ctx, id, q)
	if err == nil && n.f.fault == "start" {
		n.f.fault = ""
		err = io.ErrUnexpectedEOF
	}
	return err
}

type recoveryLostRepository struct {
	coordinator.Repository
	f *recoveryFixture
}

func (r recoveryLostRepository) Save(ctx context.Context, id string, revision uint64, c coordinator.Change) error {
	err := r.Repository.Save(ctx, id, revision, c)
	if err == nil && r.f.fault == "bind" && len(c.Next.Attempts) > 0 {
		r.f.fault = ""
		return io.ErrUnexpectedEOF
	}
	return err
}

func (f *recoveryFixture) recovery(t *testing.T) clusterapp.DiLoCoRecovery {
	t.Helper()
	// A new repository and composition on every tick simulates service restart.
	repo, err := coordinatorlocal.Open(f.repoPath)
	check(t, err)
	f.model.Repository = recoveryLostRepository{repo, f}
	inbox, err := admission.OpenInbox(f.inboxPath)
	check(t, err)
	sign := func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		if err := ctx.Err(); err != nil {
			return trust.SignedProof{}, err
		}
		return trust.SignPrincipalProof(f.x.f.a, f.x.coordinator.Chain, f.x.coordinator.Key, f.x.f.view, q, f.x.f.now)
	}
	return clusterapp.DiLoCoRecovery{Model: f.model, Node: func(a run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
		if a != f.x.prepare.Signed.Manifest.Slot.Allocation {
			return nil, fmt.Errorf("replacement allocation")
		}
		return recoveryLostNode{f.node, f}, nil
	}, Anchor: f.x.f.a, Current: func(context.Context) (trust.VerifiedSnapshot, error) { return f.x.f.view, nil }, Clock: f.x.f.clock, Sign: sign, Grant: func(ctx context.Context, request workload.SignedRequest) (workload.Grant, error) {
		return inbox.RecoveryWorkloadGrant(ctx, f.assignment.Binding.RunID, request.Request.Scope.Job, request, func() (workload.Grant, error) {
			q, e := request.GrantRequest()
			if e != nil {
				return workload.Grant{}, e
			}
			p, e := sign(ctx, q)
			return workload.Grant{Request: request, Proof: p}, e
		})
	}, Issue: func(ctx context.Context, g workload.Grant) (workload.Result, error) {
		h, _ := run.Digest(g)
		f.issuedGrants = append(f.issuedGrants, h)
		out, e := f.issuer.Issue(ctx, f.x.coordinator.Chain, g, f.x.f.view, f.x.f.now)
		if e == nil && f.fault == "issue" {
			f.fault = ""
			e = io.ErrUnexpectedEOF
		}
		return out, e
	}}
}

func TestProductionLearnerRecoveryLostResponsesAndCompletion(t *testing.T) {
	for _, stage := range []string{"authorize", "bind", "issue", "start"} {
		t.Run(stage, func(t *testing.T) {
			f := newRecoveryFixture(t)
			before := f.state(t)
			f.stop(t)
			f.fault = stage
			for tick := 0; tick < 6; tick++ {
				_, err := f.recovery(t).Tick(context.Background(), f.assignment.Binding.RunID)
				if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatal(err)
				}
				if s := f.state(t); s.Phase == coordinator.Failed {
					t.Fatal(s.Failure)
				} else if s.Recovery == nil && len(s.Attempts) == 1 {
					break
				}
				f.x.f.now = f.x.f.now.Add(5 * time.Second)
			}
			s := f.state(t)
			if s.Recovery != nil || len(s.Attempts) != 1 || f.fault != "" {
				t.Fatal("recovery did not converge", s.Recovery, len(s.Attempts), f.fault)
			}
			for _, h := range f.issuedGrants {
				if h != f.issuedGrants[0] {
					t.Fatal("clock/restart changed signed issuance bytes")
				}
			}
			if stage == "issue" && len(f.issuedGrants) != 2 {
				t.Fatal("issuance was not replayed", len(f.issuedGrants))
			}
			x, err := f.x.store.ActiveExecution(context.Background())
			check(t, err)
			if x.Learner.Stopped || x.Learner.Assignment.ExpectedPrivateHash != s.Attempts[0].PrivateHash || x.Learner.Assignment.AttemptID != s.Attempts[0].AttemptID || !reflect.DeepEqual(before.Plan.Slots, s.Plan.Slots) || !reflect.DeepEqual(before.Assignments, s.Assignments) || f.x.initializations.Load() != 2 {
				t.Fatal("retry changed identity/state or duplicated credential")
			}
			old := private.Submission{Format: "mixlab_update_submission_v1", Update: f.update, PrincipalID: x.Workload.Workload, AttemptID: f.x.prepare.Signed.Manifest.Attempt, WorkloadGeneration: 1, Allocation: s.Plan.Slots[0].Allocation}
			if _, err = f.model.AcceptSubmission(context.Background(), old); err == nil {
				t.Fatal("old attempt accepted update")
			}
			old.AttemptID = x.Workload.Attempt
			old.WorkloadGeneration = x.Workload.Generation
			for range 2 {
				_, err = f.model.AcceptSubmission(context.Background(), old)
				check(t, err)
			}
			for range 4 {
				check(t, f.model.Advance(context.Background(), s.Plan.Cohort.RunID))
			}
			s = f.state(t)
			if s.Phase != coordinator.Complete || len(s.Updates) != 1 {
				t.Fatal("recovered run did not complete exactly once", s.Phase, len(s.Updates))
			}
		})
	}
}

func TestProductionLearnerRecoveryExhaustion(t *testing.T) {
	f := newRecoveryFixture(t)
	for retry := 0; retry <= coordinator.LearnerRestartLimit; retry++ {
		f.stop(t)
		for range 4 {
			_, err := f.recovery(t).Tick(context.Background(), f.assignment.Binding.RunID)
			check(t, err)
		}
	}
	s := f.state(t)
	if s.Phase != coordinator.Failed || s.Failure == "" || len(s.Attempts) != coordinator.LearnerRestartLimit {
		t.Fatal("exhaustion invisible", s.Phase, s.Failure, len(s.Attempts))
	}
}

func TestLearnerRecoveryHTTPRequiresExactCoordinatorAndClosedSchemas(t *testing.T) {
	f := newRecoveryFixture(t)
	ctx := context.Background()
	job := f.x.prepare.Signed.Manifest.Slot.Allocation.JobID
	for _, identity := range []managedtls.Identity{f.x.f.client, f.x.f.leaf(t, trust.Coordinator, id(t))} {
		c, _ := f.x.client(t, identity)
		if _, err := c.LearnerStatus(ctx, job); err == nil {
			t.Fatal("foreign principal read learner recovery status")
		}
		if _, err := c.AuthorizeLearnerRetry(ctx, job, clusterapp.NodeLearnerRetryRequest{Attempt: id(t), PrivateHash: f.assignment.Contract.ConfigHash}); err == nil {
			t.Fatal("foreign principal authorized learner retry")
		}
		if err := c.StartLearnerRetry(ctx, job, clusterapp.NodeLearnerRetryStart{}); err == nil {
			t.Fatal("foreign principal started learner retry")
		}
	}
	if _, err := f.node.AuthorizeLearnerRetry(ctx, job, clusterapp.NodeLearnerRetryRequest{Attempt: id(t), PrivateHash: f.assignment.Contract.ConfigHash}); err == nil {
		t.Fatal("live learner replaced")
	}
	_, raw := f.x.client(t, f.x.coordinator)
	for _, operation := range []string{"retry-authorize", "retry-start"} {
		code, _ := learnerHTTPRequest(t, raw, http.MethodPut, f.x.url+"/v1/agent/learners/"+job+"/"+operation, "application/json", []byte(`{"private_path":"/foreign","credential_envelope":{}}`))
		if code != http.StatusBadRequest {
			t.Fatal("open recovery schema", operation, code)
		}
	}
	status, err := f.node.LearnerStatus(ctx, job)
	check(t, err)
	if status.Recovery != nil || status.Stopped {
		t.Fatal("rejected request mutated recovery")
	}
}
