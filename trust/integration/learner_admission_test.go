package integration

import (
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/recoveryauth"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestLearnerAdmissionCredentialHandoffAndFencing(t *testing.T) {
	t.Run("live_handoff", func(t *testing.T) { learnerAdmissionLifecycle(t, false) })
	t.Run("expired_handoff", func(t *testing.T) { learnerAdmissionLifecycle(t, true) })
}
func learnerAdmissionLifecycle(t *testing.T, expiredHandoff bool) {
	f := newTLSFixture(t)
	ctx := context.Background()
	nodeDir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	s, _, _, requirements := capabilityNodeIn(t, f, nodeDir)
	nodePath, err := statehome.Resolve(statehome.Options{ExactDir: nodeDir}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Worker)
	f.signSnapshot(t)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	caps, err := s.Capabilities(ctx, actor, f.now)
	check(t, err)
	coordinatorID := id(t)
	coordinatorIdentity := f.leaf(t, trust.Coordinator, coordinatorID)
	coordinatorActor, err := trust.AuthenticatePrincipal(f.a, f.view, coordinatorIdentity.Chain, f.now)
	check(t, err)
	reserve := nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}
	lease, err := s.Reserve(ctx, actor, reserve, f.now)
	check(t, err)
	var plan run.Plan
	b, err := os.ReadFile("../../diloco/testdata/plan.json")
	check(t, err)
	check(t, json.Unmarshal(b, &plan))
	member := distributed.LearnerMember{WorkerID: id(t), ParticipantID: caps.Node, IslandID: "single-metal", DatasetPartitionID: "partition0"}
	plan.Cohort, err = distributed.NewLearnerCohort(reserve.Run, 1, []distributed.LearnerMember{member})
	check(t, err)
	allocation := run.Allocation{Provider: "node_agent", NodeID: caps.Node, LeaseID: lease.ID, JobID: id(t)}
	allocation.RefHash, err = run.Digest(allocation)
	check(t, err)
	capHash, err := run.Digest(caps)
	check(t, err)
	plan.Slots = []run.Slot{{Member: member, CapabilityHash: capHash, Allocation: allocation}}
	plan.Coordinator = run.Coordinator{ClusterID: f.a.Cluster(), PrincipalID: coordinatorID, Role: "coordinator", Audience: "learner-run"}
	plan.CreatedAt = f.now.Unix()
	plan.ExpiresAt = f.now.Add(time.Minute).Unix()
	config := json.RawMessage(`{"model_dim":16}`)
	plan.Contract.ConfigHash = nodejob.Hash(config)
	plan.SHA256, err = plan.Digest()
	check(t, err)
	m := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: actor.Principal, Attempt: id(t), WorkloadGeneration: 1, Plan: plan, Slot: plan.Slots[0], Config: config, DatasetSelector: "toy", Limits: requirements.Limits}
	q, err := m.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	signed := nodejob.SignedLearner{Manifest: m, Proof: proof}
	for _, mutate := range []func(*nodejob.SignedLearner){func(j *nodejob.SignedLearner) { j.Manifest.Attempt = id(t) }, func(j *nodejob.SignedLearner) { j.Manifest.Slot.Member.WorkerID = id(t) }, func(j *nodejob.SignedLearner) { j.Proof.Request.Audience = id(t) }} {
		bad := signed
		mutate(&bad)
		if _, e := nodejob.AcceptLearner(f.a, f.view, actor, caps.Node, bad, f.now); e == nil {
			t.Fatal("tampered learner accepted")
		}
	}
	accepted, err := nodejob.AcceptLearner(f.a, f.view, actor, caps.Node, signed, f.now)
	check(t, err)
	prep := nodeagent.PreparationPorts{Clock: f.clock, Probe: func(context.Context) (workerprobe.Report, error) { return caps.Probe, nil }, DatasetID: func(context.Context, string) (string, error) { return strings.Repeat("b", 64), nil }}
	if _, e := s.PrepareLearnerChecked(ctx, actor, accepted, lease.Version, prep); e == nil {
		t.Fatal("prepared changed dataset contents")
	}
	prep.DatasetID = func(_ context.Context, path string) (string, error) {
		if path != "/private/local/train.bin" {
			t.Fatal("dataset path did not come from local catalog")
		}
		return requirements.DatasetID, nil
	}
	beforePrepare, err := os.ReadFile(filepath.Join(nodeDir, "node-leases.json"))
	check(t, err)
	job, err := s.PrepareLearnerChecked(ctx, actor, accepted, lease.Version, prep)
	check(t, err)
	// Recreate the durable state after the job write, before the lease write.
	check(t, os.WriteFile(filepath.Join(nodeDir, "node-leases.json"), beforePrepare, 0600))
	s, err = nodeagent.Open(nodePath, f.a.Cluster(), caps.Node)
	check(t, err)
	_, err = s.PrepareLearner(ctx, actor, accepted, lease.Version, f.now)
	check(t, err)
	preparedLease, err := s.LeaseStatus(ctx, actor, lease.ID, f.now)
	check(t, err)
	if preparedLease.Expires != plan.ExpiresAt {
		t.Fatal("prepared lease outlives admission")
	}
	if _, e := s.RenewLease(ctx, actor, nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: lease.ID, ExpectedVersion: preparedLease.Version, TTLSeconds: 300}, f.now.Add(5*time.Second)); e == nil {
		t.Fatal("controller renewed learner beyond admission deadline")
	}
	if _, e := s.StartLearner(ctx, actor, allocation.JobID, id(t), f.now); e == nil {
		t.Fatal("started before credential preparation")
	}
	nodeKey := key(t)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	der, err := certificates.IssueNode(f.a, f.issuer, f.issuerKey, caps.Node, nodeKey.Public(), envelope.PublicKey().Bytes(), f.now)
	check(t, err)
	nodeChain := [][]byte{der, f.issuer, f.a.DER()}
	var request workload.SignedRequest
	initializations := 0
	wp := nodeagent.WorkloadPorts{Clock: f.clock, Initialize: func(context.Context, workload.Scope) error { initializations++; return nil }, Request: func(_ context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		if request.Request.Version != "" {
			return request, nil
		}
		r, e := workload.NewRequest(scope, key(t))
		if e != nil {
			return request, e
		}
		sq, e := r.SigningRequest()
		if e != nil {
			return request, e
		}
		p, e := trust.SignPrincipalProof(f.a, nodeChain, nodeKey, f.view, sq, f.now)
		request = workload.SignedRequest{Request: r, Proof: p}
		return request, e
	}}
	request, err = s.PrepareLearnerWorkload(ctx, actor, allocation.JobID, wp)
	check(t, err)
	_, err = s.PrepareLearnerWorkload(ctx, actor, allocation.JobID, wp)
	check(t, err)
	if initializations != 1 || request.Request.Scope.Group != "" || request.Request.Scope.Worker != member.WorkerID {
		t.Fatal("incorrect workload preparation")
	}
	grantQ, err := request.GrantRequest()
	check(t, err)
	grantProof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, grantQ, f.now)
	check(t, err)
	grant := workload.Grant{Request: request, Proof: grantProof}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	path, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Authority})
	check(t, err)
	issuer, err := workload.Initialize(ctx, path, workload.Authority{Anchor: f.a, Issuer: f.issuer, Key: f.issuerKey}, f.view, f.now)
	check(t, err)
	credential, err := issuer.Issue(ctx, f.client.Chain, grant, f.view, f.now)
	check(t, err)
	check(t, s.ActivateLearnerCredential(ctx, actor, allocation.JobID, f.a, grant, credential, f.now))
	startKey := id(t)
	assignment, err := s.StartLearner(ctx, actor, allocation.JobID, startKey, f.now)
	check(t, err)
	again, err := s.StartLearner(ctx, actor, allocation.JobID, startKey, f.now)
	check(t, err)
	aHash, _ := assignment.Binding()
	bHash, _ := again.Binding()
	if aHash != bHash {
		t.Fatal("duplicate start changed binding")
	}
	x, err := s.ActiveExecution(ctx)
	check(t, err)
	if x == nil || x.Learner == nil || x.Approval == nil || x.Transport != nil || x.Approval.Learner == nil {
		t.Fatal("learner did not resolve to its own hosting variant")
	}
	check(t, s.CheckExecutionTrust(ctx, f.a, f.view, f.now))
	approvalHash, err := run.Digest(*x.Approval)
	check(t, err)
	started := contract.Outcome{JobID: assignment.JobID, AttemptID: assignment.AttemptID, ManifestHash: q.Digest, ApprovalHash: approvalHash, Version: 1, Kind: contract.Started, PID: 1234}
	check(t, s.ApplyLearnerOutcome(ctx, started))
	check(t, s.ApplyLearnerOutcome(ctx, started))
	precommitLease, err := s.LeaseStatus(ctx, actor, lease.ID, f.now)
	check(t, err)
	if precommitLease.State != nodeagent.Prepared {
		t.Fatal("process start bypassed coordinator commit")
	}
	var reg run.Registration
	b, err = os.ReadFile("../../diloco/testdata/registration.json")
	check(t, err)
	check(t, json.Unmarshal(b, &reg))
	reg.RunID = plan.Cohort.RunID
	reg.PlanHash = plan.SHA256
	reg.Generation = plan.Cohort.Generation
	reg.CohortHash = plan.Cohort.CohortHash
	reg.Slot = m.Slot
	reg.Contract = plan.Contract
	reg.BaseModelHash = plan.BaseModel.Ref.SHA256
	reg.Coordinator = plan.Coordinator
	reg.AttemptID = m.Attempt
	reg.JobManifestHash = q.Digest
	commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: plan.Cohort.RunID, PlanHash: plan.SHA256, Generation: plan.Cohort.Generation, CohortHash: plan.Cohort.CohortHash}, Coordinator: plan.Coordinator, Registrations: []run.Registration{reg}, CommittedAt: f.now.Unix()}
	commit.Binding.CommitHash, err = commit.Digest()
	check(t, err)
	check(t, commit.ValidateFor(plan))
	cq := trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: commit.Binding.CommitHash, Context: plan.Cohort.RunID, Audience: plan.Coordinator.Audience}
	cp, err := trust.SignPrincipalProof(f.a, coordinatorIdentity.Chain, coordinatorIdentity.Key, f.view, cq, f.now)
	check(t, err)
	c := authenticated.Commit{Body: commit, Proof: cp}
	handoffNow := f.now
	if expiredHandoff {
		handoffNow = time.Unix(lease.Expires, 0)
	}
	beforeHandoff, err := os.ReadFile(filepath.Join(nodeDir, "node-leases.json"))
	check(t, err)
	check(t, s.FinalizeLearner(ctx, actor, allocation.JobID, f.a, f.view, c, handoffNow))
	// Commit evidence is durable but the lease publication was interrupted.
	check(t, os.WriteFile(filepath.Join(nodeDir, "node-leases.json"), beforeHandoff, 0600))
	s, err = nodeagent.Open(nodePath, f.a.Cluster(), caps.Node)
	check(t, err)
	if _, e := s.AbortReservation(ctx, actor, reserve, f.now); e == nil {
		t.Fatal("interrupted handoff restored submitting controller authority")
	}
	check(t, s.FinalizeLearner(ctx, coordinatorActor, allocation.JobID, f.a, f.view, c, handoffNow))
	live, err := s.LeaseStatus(ctx, coordinatorActor, lease.ID, f.now)
	check(t, err)
	if expiredHandoff {
		if live.State != nodeagent.Releasing || live.CleanupReason != "expired" || live.Coordinator != coordinatorID {
			t.Fatal("expired handoff revived allocation")
		}
		check(t, s.ConfirmLearnerStopped(ctx, aHash))
		check(t, s.CleanupLearner(ctx, allocation.JobID, func(context.Context, workload.Scope) error { return nil }))
		av, e := s.Availability(handoffNow)
		check(t, e)
		if !av.Available {
			t.Fatal("expired handoff stranded allocation")
		}
		return
	}
	if live.Coordinator != coordinatorID || live.State != nodeagent.Running {
		t.Fatal("lease not handed off")
	}
	check(t, s.CheckExecutionTrust(ctx, f.a, f.view, f.now))
	nodePolicy, err := managedtls.New(managedtls.Identity{Chain: nodeChain, Key: nodeKey}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}, f.clock)
	check(t, err)
	clientPolicy, err := managedtls.New(coordinatorIdentity, f.verify(trust.Node, caps.Node), f.clock)
	check(t, err)
	handler, err := clusterapp.NodeManagementHandler(s, nodePolicy, f.clock, &nodeHTTPJobs{})
	check(t, err)
	server := serveCapabilities(t, nodePolicy, handler)
	client, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(server.URL, "https://"), caps.Node, func() (*managedtls.Policy, error) { return clientPolicy, nil })
	check(t, err)
	remote := clusterapp.LearnerAllocations{Client: func(string) (*clusterapp.NodeClient, error) { return client, nil }}
	remoteLease, err := remote.Status(ctx, allocation)
	check(t, err)
	if remoteLease != live {
		t.Fatal("coordinator HTTP lease status changed")
	}
	if _, e := client.Reserve(ctx, reserve); e == nil {
		t.Fatal("coordinator acquired controller reservation authority")
	}
	command := nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: live.ID, ExpectedVersion: live.Version, TTLSeconds: 300}
	if _, e := s.RenewLease(ctx, actor, command, f.now.Add(20*time.Second)); e == nil {
		t.Fatal("submitter renewed after handoff")
	}
	if _, e := s.AbortReservation(ctx, actor, reserve, f.now); e == nil {
		t.Fatal("submitter compensated committed allocation")
	}
	live, err = s.RenewLease(ctx, coordinatorActor, command, f.now.Add(20*time.Second))
	check(t, err)
	// Lease authority is the committed coordinator principal, not the role.
	foreign := coordinatorActor
	foreign.Principal = id(t)
	if _, e := s.LeaseStatus(ctx, foreign, live.ID, f.now); e == nil {
		t.Fatal("foreign coordinator read handed-off lease")
	}
	if _, e := s.RenewLease(ctx, foreign, nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: live.ID, ExpectedVersion: live.Version, TTLSeconds: 300}, f.now.Add(25*time.Second)); e == nil {
		t.Fatal("foreign coordinator renewed handed-off lease")
	}
	if _, e := s.AuthorizeLearnerRetry(ctx, foreign, allocation.JobID, id(t), strings.Repeat("d", 64), f.now); e == nil {
		t.Fatal("foreign coordinator authorized retry")
	}
	if _, e := s.AuthorizeLearnerRetry(ctx, coordinatorActor, allocation.JobID, id(t), strings.Repeat("d", 64), f.now); e == nil {
		t.Fatal("live child was replaced")
	}
	wrong := aHash
	wrong.AttemptID = id(t)
	if s.ConfirmLearnerStopped(ctx, wrong) == nil {
		t.Fatal("foreign child outcome accepted")
	}
	failed := started
	failed.Version, failed.Kind, failed.NoChild, failed.Error = 2, contract.Failed, true, "synthetic child failure"
	check(t, s.ApplyLearnerOutcome(ctx, failed))
	check(t, s.ApplyLearnerOutcome(ctx, failed))
	if s.ApplyLearnerOutcome(ctx, started) == nil {
		t.Fatal("outcome rollback accepted")
	}
	afterFailure, err := s.LeaseStatus(ctx, coordinatorActor, lease.ID, f.now)
	check(t, err)
	if afterFailure.State != nodeagent.Running {
		t.Fatal("child failure surrendered recovery allocation")
	}
	retry, err := s.AuthorizeLearnerRetry(ctx, coordinatorActor, allocation.JobID, id(t), strings.Repeat("d", 64), f.now)
	check(t, err)
	rq, err := retry.SigningRequest()
	check(t, err)
	rp, err := trust.SignPrincipalProof(f.a, nodeChain, nodeKey, f.view, rq, f.now)
	check(t, err)
	check(t, (nodejob.SignedLearnerRecovery{Body: retry, Proof: rp}).Verify(f.a, f.view, f.now))
	auth := recoveryauth.Authorization{Anchor: f.a, Current: func(context.Context) (trust.VerifiedSnapshot, error) { return f.view, nil }, Clock: f.clock, Coordinator: plan.Coordinator, Evidence: nodejob.SignedLearnerRecovery{Body: retry, Proof: rp}, Confirm: func(context.Context, nodejob.LearnerRecovery) error { return nil }}
	check(t, auth.Check(ctx, retry.Attempt))
	otherAttempt := retry.Attempt
	otherAttempt.AttemptID = id(t)
	if auth.Check(ctx, otherAttempt) == nil {
		t.Fatal("recovery adapter accepted another attempt")
	}
	auth.Confirm = func(context.Context, nodejob.LearnerRecovery) error { return errors.New("lease expired") }
	if auth.Check(ctx, retry.Attempt) == nil {
		t.Fatal("signed proof bypassed current node fence")
	}
	if retry.Attempt.Allocation != allocation || retry.Attempt.PreviousAttemptID != m.Attempt {
		t.Fatal("retry changed allocation")
	}
	request = workload.SignedRequest{}
	request, err = s.PrepareLearnerRetryWorkload(ctx, coordinatorActor, allocation.JobID, wp)
	check(t, err)
	if request.Request.Scope.Controller != coordinatorID || request.Request.Scope.Attempt != retry.Attempt.AttemptID || request.Request.Scope.Generation != 2 {
		t.Fatal("retry retained original grant authority")
	}
	grantQ, err = request.GrantRequest()
	check(t, err)
	grantProof, err = trust.SignPrincipalProof(f.a, coordinatorIdentity.Chain, coordinatorIdentity.Key, f.view, grantQ, f.now)
	check(t, err)
	grant = workload.Grant{Request: request, Proof: grantProof}
	credential, err = issuer.Issue(ctx, coordinatorIdentity.Chain, grant, f.view, f.now)
	check(t, err)
	ah, err := run.Digest(retry.Attempt)
	check(t, err)
	aq := trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RecoveryGrant, Digest: ah, Context: plan.Cohort.RunID, Audience: plan.Coordinator.Audience}
	ap, err := trust.SignPrincipalProof(f.a, coordinatorIdentity.Chain, coordinatorIdentity.Key, f.view, aq, f.now)
	check(t, err)
	ack := authenticated.Recovery{Body: retry.Attempt, Coordinator: plan.Coordinator, Proof: ap}
	cleanupCount := 0
	restarted, err := s.StartLearnerRetry(ctx, coordinatorActor, allocation.JobID, f.a, f.view, ack, grant, credential, func(_ context.Context, old workload.Scope) error {
		cleanupCount++
		if old.Attempt != m.Attempt {
			t.Fatal("wrong prior key cleanup")
		}
		return nil
	}, f.now)
	check(t, err)
	if restarted.Member != assignment.Member || restarted.PlanHash != assignment.PlanHash || restarted.AttemptID == assignment.AttemptID || cleanupCount != 1 {
		t.Fatal("retry changed semantic identity or reused attempt")
	}
	x, err = s.ActiveExecution(ctx)
	check(t, err)
	if x.Outcome != nil || x.Learner.Stopped || x.Approval.AttemptID() != restarted.AttemptID {
		t.Fatal("retry retained old hosting evidence")
	}
	if s.ApplyLearnerOutcome(ctx, failed) == nil {
		t.Fatal("old hosting outcome fenced retry")
	}
	_, err = s.StartLearnerRetry(ctx, coordinatorActor, allocation.JobID, f.a, f.view, ack, grant, credential, func(context.Context, workload.Scope) error {
		t.Fatal("exact retry cleaned another credential")
		return nil
	}, f.now)
	check(t, err)
	if s.ConfirmLearnerStopped(ctx, aHash) == nil {
		t.Fatal("old outcome fenced new child")
	}
	newBinding, err := restarted.Binding()
	check(t, err)
	check(t, s.ConfirmLearnerStopped(ctx, newBinding))
	second, err := s.AuthorizeLearnerRetry(ctx, coordinatorActor, allocation.JobID, id(t), strings.Repeat("e", 64), f.now)
	check(t, err)
	if second.Attempt.WorkloadGeneration != 3 || second.Attempt.PreviousAttemptID != restarted.AttemptID {
		t.Fatal("retry history not advanced")
	}
	check(t, s.Expire(ctx, time.Unix(live.Expires, 0)))
	if _, e := s.AuthorizeLearnerRetry(ctx, coordinatorActor, allocation.JobID, retry.Attempt.AttemptID, retry.Attempt.PrivateHash, time.Unix(live.Expires, 0)); e == nil {
		t.Fatal("expired allocation authorized retry")
	}
	cleaned := false
	check(t, s.CleanupLearner(ctx, allocation.JobID, func(_ context.Context, scope workload.Scope) error {
		cleaned = true
		if scope.Workload != request.Request.Scope.Workload {
			t.Fatal("wrong key cleanup")
		}
		return nil
	}))
	available, err := s.Availability(time.Unix(live.Expires, 0))
	check(t, err)
	if !available.Available || !cleaned || job.Version != 1 {
		t.Fatal("stranded accelerator")
	}
}
