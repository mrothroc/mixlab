//go:build darwin || linux

package clusterapp

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestNodeLearnerPollStaleTerminalAfterRetry(t *testing.T) {
	f := newLearnerRuntimeFixture(t)
	ctx := context.Background()
	s := f.remote.runtime.Store
	x, err := s.ActiveExecution(ctx)
	check(t, err)
	m := x.Learner.Signed.Manifest
	a := f.authority
	v, err := a.Current(ctx, testNow)
	check(t, err)
	controller, _, err := f.controller.Active(testNow)
	check(t, err)
	submitter, err := trust.AuthenticatePrincipal(a.Anchor, v, controller.Chain, testNow)
	check(t, err)
	identity, signer, err := f.coordinator.Active(testNow)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(a.Anchor, v, identity.Chain, testNow)
	check(t, err)
	sign := func(q trust.SignRequest) trust.SignedProof {
		t.Helper()
		p, err := trust.SignPrincipalProof(a.Anchor, identity.Chain, signer, v, q, testNow)
		check(t, err)
		return p
	}

	var reg run.Registration
	b, err := os.ReadFile("../../diloco/testdata/registration.json")
	check(t, err)
	check(t, json.Unmarshal(b, &reg))
	reg.RunID, reg.PlanHash = m.Plan.Cohort.RunID, m.Plan.SHA256
	reg.Generation, reg.CohortHash = m.Plan.Cohort.Generation, m.Plan.Cohort.CohortHash
	reg.Slot, reg.Contract = m.Slot, m.Plan.Contract
	reg.BaseModelHash, reg.Coordinator = m.Plan.BaseModel.Ref.SHA256, m.Plan.Coordinator
	reg.AttemptID, reg.JobManifestHash = m.Attempt, x.Approval.ManifestHash
	c := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: reg.RunID, PlanHash: reg.PlanHash, Generation: reg.Generation, CohortHash: reg.CohortHash}, Coordinator: reg.Coordinator, Registrations: []run.Registration{reg}, CommittedAt: testNow.Unix()}
	c.Binding.CommitHash, err = c.Digest()
	check(t, err)
	cp := sign(trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: c.Binding.CommitHash, Context: reg.RunID, Audience: reg.Coordinator.Audience})
	check(t, s.FinalizeLearner(ctx, submitter, x.Job.ID, a.Anchor, v, authenticated.Commit{Body: c, Proof: cp}, testNow))

	approvalHash, err := run.Digest(*x.Approval)
	check(t, err)
	terminal := contract.Outcome{JobID: x.Job.ID, AttemptID: m.Attempt, ManifestHash: x.Approval.ManifestHash, ApprovalHash: approvalHash, Version: 1, Kind: contract.Failed, NoChild: true, Error: "stopped learner"}
	check(t, s.ApplyLearnerOutcome(ctx, terminal))
	stale, err := s.ActiveExecution(ctx)
	check(t, err)
	loop := nodeExecutionLoop{store: s}
	check(t, loop.pollLearner(ctx, *stale, false))

	next, err := controllerRandomID()
	check(t, err)
	retry, err := s.AuthorizeLearnerRetry(ctx, actor, x.Job.ID, next, strings.Repeat("c", 64), testNow)
	check(t, err)
	request, err := s.PrepareLearnerRetryWorkload(ctx, actor, x.Job.ID, f.ports)
	check(t, err)
	q, err := request.GrantRequest()
	check(t, err)
	grant := workload.Grant{Request: request, Proof: sign(q)}
	credential, err := a.Workloads.Issue(ctx, identity.Chain, grant, v, testNow)
	check(t, err)
	digest, err := run.Digest(retry.Attempt)
	check(t, err)
	ack := authenticated.Recovery{Body: retry.Attempt, Coordinator: reg.Coordinator, Proof: sign(trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RecoveryGrant, Digest: digest, Context: reg.RunID, Audience: reg.Coordinator.Audience})}
	_, err = s.StartLearnerRetry(ctx, actor, x.Job.ID, a.Anchor, v, ack, grant, credential, func(context.Context, workload.Scope) error { return nil }, testNow)
	check(t, err)
	before, err := s.ActiveExecution(ctx)
	check(t, err)
	if before.Approval.AttemptID() != next || before.Outcome != nil || before.Learner.Stopped {
		t.Fatal("retry did not publish a live replacement assignment")
	}
	// Deterministic interleaving: polling read the terminal snapshot before the
	// recovery HTTP handler installed the replacement, then resumes afterward.
	if err := loop.pollLearner(ctx, *stale, false); err != nil {
		t.Fatalf("stale terminal poll would stop the agent: %v", err)
	}
	if err := s.ApplyLearnerOutcome(ctx, terminal); err == nil {
		t.Fatal("actual old-attempt publication bypassed the strict outcome fence")
	}
	after, err := s.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) || loop.running != nil {
		t.Fatal("stale terminal poll changed or launched the replacement attempt")
	}
}
