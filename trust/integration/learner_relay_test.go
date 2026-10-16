package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type relayClock struct{ f *tlsFixture }

func (c relayClock) Unix() int64 { return c.f.clock().Unix() }

type relayAllocations struct{}

func (relayAllocations) Ready(context.Context, run.Slot) error { return nil }

type relayEvaluator struct{}

func (relayEvaluator) Evaluate(context.Context, run.EvaluationAssignment) (run.EvaluationResult, error) {
	return run.EvaluationResult{}, fmt.Errorf("not submitted")
}

type relayFixture struct {
	*artifactFixture
	model      coordinator.Model
	url        string
	bindings   []trust.WorkloadBinding
	identities []managedtls.Identity
	options    []clusterapp.LearnerClientOptions
	clients    []*clusterapp.LearnerClient
	body       []byte
}

func newLearnerRelay(t *testing.T) *relayFixture {
	return newLearnerRelayModel(t, []optimizer.Tensor{{Name: "weight", Shape: []int{1}, Values: []float32{1}}}, precision.Policy{ComputeDType: "float32", Backend: "metal"})
}

func newLearnerRelayModel(t *testing.T, model []optimizer.Tensor, precisionPolicy precision.Policy) *relayFixture {
	t.Helper()
	ctx := context.Background()
	var body bytes.Buffer
	check(t, tensors.Write(&body, model))
	precisionHash, err := precisionPolicy.Digest()
	check(t, err)
	a := newArtifactFixture(t, artifactRef(body.Bytes()))
	f := a.f
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Worker)
	f.signSnapshot(t)
	p := a.s.Plan
	for i := range p.Slots {
		p.Slots[i].Member.WorkerID = id(t)
		p.Slots[i].Allocation.JobID = id(t)
		p.Slots[i].Allocation.LeaseID = id(t)
		p.Slots[i].Allocation.RefHash = ""
		p.Slots[i].Allocation.RefHash, _ = run.Digest(p.Slots[i].Allocation)
	}
	p.Cohort, err = distributed.NewLearnerCohort(id(t), 1, []distributed.LearnerMember{p.Slots[0].Member, p.Slots[1].Member})
	check(t, err)
	p.SHA256, err = p.Digest()
	check(t, err)
	commit := *a.s.Commit
	commit.Binding = run.Binding{RunID: p.Cohort.RunID, PlanHash: p.SHA256, Generation: 1, CohortHash: p.Cohort.CohortHash}
	for i := range commit.Registrations {
		r := &commit.Registrations[i]
		r.RunID = p.Cohort.RunID
		r.PlanHash = p.SHA256
		r.Generation = 1
		r.CohortHash = p.Cohort.CohortHash
		r.Slot = p.Slots[i]
		r.AttemptID = id(t)
		r.WorkloadGeneration = 1
		r.Precision = precisionPolicy
		r.PrecisionHash = precisionHash
	}
	commit.Binding.CommitHash, err = commit.Digest()
	check(t, err)
	a.s.Plan = p
	a.s.Commit = &commit
	a.s.Policy.WorkerID = p.Slots[0].Member.WorkerID
	for i := range a.s.Assignments {
		a.s.Assignments[i].Binding = commit.Binding
		a.s.Assignments[i].WorkerID = p.Slots[i].Member.WorkerID
		a.s.Assignments[i].PrecisionHash = precisionHash
	}
	check(t, a.g.Repository.Save(ctx, p.Cohort.RunID, 0, coordinator.Change{Next: a.s}))
	check(t, a.g.Store.PutVerified(ctx, p.BaseModel.Ref, bytes.NewReader(body.Bytes())))
	x := &relayFixture{artifactFixture: a, body: body.Bytes()}
	x.model = coordinator.Model{Repository: a.g.Repository, Artifacts: coordinatorlocal.Artifacts{Store: a.g.Store}, Allocations: relayAllocations{}, Clock: relayClock{f}, Evaluator: relayEvaluator{}}
	policy, err := learnerapi.NewPolicy(f.server, f.a, artifactTrust{f}, f.clock)
	check(t, err)
	a.g.Policy = policy
	grants, err := learnerapi.OpenGrants(artifactPath(t))
	check(t, err)
	g := learnerapi.Gateway{Grants: grants, Policy: policy, Anchor: f.a, Trust: artifactTrust{f}, Coordinator: p.Coordinator, Model: x.model, Clock: f.clock, Timeout: 10 * time.Second, Sign: func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return trust.SignPrincipalProof(f.a, f.server.Chain, f.server.Key, f.view, q, f.now)
	}}
	h, err := g.Handler()
	check(t, err)
	artifacts, err := a.g.Handler()
	check(t, err)
	mux := http.NewServeMux()
	mux.Handle(learnerapi.Route, h)
	mux.Handle("/v1/diloco/artifacts/", artifacts)
	x.url = serveCapabilities(t, policy, mux).URL
	for i, r := range commit.Registrations {
		b := trust.WorkloadBinding{Version: trust.LearnerBindingVersion, Cluster: f.a.Cluster(), Role: trust.Worker, Principal: id(t), Participant: r.Slot.Member.ParticipantID, Run: r.RunID, Job: r.Slot.Allocation.JobID, Lease: r.Slot.Allocation.LeaseID, ManifestHash: r.JobManifestHash, Attempt: r.AttemptID, Audience: "diloco/learner", IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(time.Hour).Unix(), Generation: 1, Worker: r.Slot.Member.WorkerID, PlanHash: p.SHA256, Coordinator: f.serverID}
		identity := x.issue(t, b)
		workloadPolicy, err := managedtls.New(identity, f.verify(trust.Coordinator, f.serverID), f.clock)
		check(t, err)
		nodePolicy, err := managedtls.New(a.clients[i], f.verify(trust.Coordinator, f.serverID), f.clock)
		check(t, err)
		o := clusterapp.LearnerClientOptions{Plan: p, Commit: commit, Workload: b, Endpoint: x.url, Anchor: f.a, Trust: artifactTrust{f}, Clock: f.clock, Policy: func() (*managedtls.Policy, error) { return workloadPolicy, nil }, NodePolicy: func() (*managedtls.Policy, error) { return nodePolicy, nil }}
		client, err := clusterapp.NewLearnerClientWithPolicy(o)
		check(t, err)
		x.bindings = append(x.bindings, b)
		x.identities = append(x.identities, identity)
		x.options = append(x.options, o)
		x.clients = append(x.clients, client)
	}
	return x
}
func (x *relayFixture) issue(t *testing.T, b trust.WorkloadBinding) managedtls.Identity {
	t.Helper()
	k := key(t)
	f := x.f
	der, err := certificates.IssueWorkload(f.a, f.issuer, f.issuerKey, k.Public(), b, time.Unix(b.ExpiresAt, 0), f.now)
	check(t, err)
	return managedtls.Identity{Chain: [][]byte{der, f.issuer, f.a.DER()}, Key: k}
}
func (x *relayFixture) raw(t *testing.T, identity managedtls.Identity, q learnerapi.Request) int {
	t.Helper()
	p, err := managedtls.New(identity, x.f.verify(trust.Coordinator, x.f.serverID), x.f.clock)
	check(t, err)
	c, err := p.HTTPClient(10 * time.Second)
	check(t, err)
	defer c.CloseIdleConnections()
	b, err := json.Marshal(q)
	check(t, err)
	r, err := c.Post(x.url+learnerapi.Route, "application/json", bytes.NewReader(b))
	if err != nil {
		return 0
	}
	defer func() { _ = r.Body.Close() }()
	_, _ = io.Copy(io.Discard, r.Body)
	return r.StatusCode
}
func (x *relayFixture) pollRequest() learnerapi.Request {
	return learnerapi.Request{Workload: x.bindings[0], Call: private.RelayRequest{Format: "mixlab_learner_relay_v1", Binding: x.s.Commit.Binding, WorkerID: x.bindings[0].Worker, Operation: "poll"}}
}

func TestLearnerRelayTLSRoundsAndReceipts(t *testing.T) {
	x := newLearnerRelay(t)
	ctx := context.Background()
	for round := range 2 {
		for i, c := range x.clients {
			r, err := c.Poll(ctx)
			check(t, err)
			if r.Assignment == nil || r.Assignment.Round != uint64(round) {
				t.Fatal("missing assignment")
			}
			a := *r.Assignment
			var bytesOut bytes.Buffer
			check(t, c.Read(ctx, a, a.BaseModel, &bytesOut))
			h, err := run.Digest(a)
			check(t, err)
			start := private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: h, PrivateHash: strings.Repeat("c", 64)}
			check(t, c.Start(ctx, start))
			check(t, c.Start(ctx, start))
			delta := run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}
			if c.Write(ctx, a, delta, bytes.NewReader(x.body[:3])) == nil {
				t.Fatal("truncated upload accepted")
			}
			check(t, c.Write(ctx, a, delta, bytes.NewReader(x.body)))
			x.f.mu.Lock()
			x.f.now = x.f.now.Add(time.Second)
			x.f.mu.Unlock()
			check(t, c.Write(ctx, a, delta, bytes.NewReader(x.body)))
			u := private.Update{Format: "mixlab_outer_update_v1", ID: fmt.Sprintf("update-%d-%d", round, i), AssignmentHash: h, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: x.bindings[i].Participant, IslandID: x.s.Plan.Slots[i].Member.IslandID, Contract: a.Contract, Delta: delta, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: strings.Repeat("d", 64), Attempted: a.InnerSteps, Committed: a.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 1, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: x.f.now.Unix(), FinishedAt: x.f.now.Unix()}
			first, err := c.Submit(ctx, u)
			check(t, err)
			retry, err := c.Submit(ctx, u)
			check(t, err)
			lookup, err := c.Receipt(ctx, u)
			check(t, err)
			if first != retry || first != lookup {
				t.Fatal("lost response changed receipt")
			}
			u.Loss = 1
			if _, err = c.Receipt(ctx, u); err == nil {
				t.Fatal("conflicting receipt accepted")
			}
		}
		for range 3 {
			check(t, x.model.Advance(ctx, x.s.Plan.Cohort.RunID))
		}
		r, err := x.clients[0].Poll(ctx)
		check(t, err)
		if r.Evaluation == nil {
			t.Fatal("evaluation not relayed")
		}
		a := *r.Evaluation
		var evaluated bytes.Buffer
		check(t, x.clients[0].ReadEvaluation(ctx, a, &evaluated))
		if err := x.clients[1].ReadEvaluation(ctx, a, &bytes.Buffer{}); err == nil {
			t.Fatal("non-evaluator read evaluation model")
		}
		h, _ := run.Digest(a)
		check(t, x.clients[0].Evaluate(ctx, run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: a, AssignmentHash: h, Loss: 1, EffectiveTokens: 8, StartedAt: x.f.now.Unix(), FinishedAt: x.f.now.Unix(), PrivateBeforeHash: strings.Repeat("d", 64), PrivateAfterHash: strings.Repeat("d", 64)}))
		if round == 0 {
			_, err = x.model.OpenRound(ctx, x.s.Plan.Cohort.RunID, 600)
			check(t, err)
		}
	}
	r, err := x.clients[0].Poll(ctx)
	check(t, err)
	if !r.Done {
		t.Fatal("completion not relayed")
	}
}

func TestLearnerRelayTLSBindingAndTrust(t *testing.T) {
	x := newLearnerRelay(t)
	ctx := context.Background()
	q := x.pollRequest()
	if code := x.raw(t, x.identities[0], q); code != 200 {
		t.Fatal(code)
	}
	for name, change := range map[string]func(*trust.WorkloadBinding){
		"attempt": func(b *trust.WorkloadBinding) { b.Attempt = id(t) }, "generation": func(b *trust.WorkloadBinding) { b.Generation++ },
		"manifest": func(b *trust.WorkloadBinding) { b.ManifestHash = strings.Repeat("e", 64) }, "worker": func(b *trust.WorkloadBinding) { b.Worker = id(t) },
		"run": func(b *trust.WorkloadBinding) { b.Run = id(t) }, "plan": func(b *trust.WorkloadBinding) { b.PlanHash = strings.Repeat("e", 64) },
		"coordinator": func(b *trust.WorkloadBinding) { b.Coordinator = id(t) }, "node": func(b *trust.WorkloadBinding) { b.Participant = id(t) },
		"job": func(b *trust.WorkloadBinding) { b.Job = id(t) }, "lease": func(b *trust.WorkloadBinding) { b.Lease = id(t) },
	} {
		t.Run(name, func(t *testing.T) {
			bad := q
			change(&bad.Workload)
			if code := x.raw(t, x.issue(t, bad.Workload), bad); code != 403 {
				t.Fatal(code)
			}
		})
	}
	if code := x.raw(t, x.artifactFixture.clients[0], q); code != 403 {
		t.Fatal("node certificate used for learner RPC", code)
	}
	bad := q
	bad.Workload.Principal = id(t)
	if code := x.raw(t, x.identities[0], bad); code != 403 {
		t.Fatal("body substituted TLS identity", code)
	}
	for _, failure := range []string{"revoked-worker", "revoked-server", "stale-trust"} {
		t.Run(failure, func(t *testing.T) {
			x := newLearnerRelay(t)
			_, err := x.clients[0].Poll(ctx)
			check(t, err)
			x.f.mu.Lock()
			if failure == "stale-trust" {
				x.f.now = x.f.now.Add(trust.SnapshotLifetime + time.Second)
			} else {
				principal := x.bindings[0].Principal
				if failure == "revoked-server" {
					principal = x.f.serverID
				}
				x.f.snapshot.Payload.Generation++
				x.f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: principal, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
				x.f.signSnapshot(t)
			}
			x.f.mu.Unlock()
			if _, err = x.clients[0].Poll(ctx); err == nil {
				t.Fatal("untrusted peer accepted")
			}
		})
	}
}

type relayRecovery struct{}

func (relayRecovery) Check(context.Context, run.AttemptBinding) error { return nil }
func TestLearnerRelayFencesReplacedAttempt(t *testing.T) {
	x := newLearnerRelay(t)
	ctx := context.Background()
	old := x.clients[0]
	a := x.s.Assignments[0]
	hash, _ := run.Digest(a)
	start := private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: hash, PrivateHash: strings.Repeat("c", 64)}
	check(t, old.Start(ctx, start))
	b := x.bindings[0]
	prior := b.Attempt
	b.Attempt = id(t)
	b.Generation++
	b.Principal = id(t)
	check(t, x.model.BindAttempt(ctx, run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: a.Binding, WorkerID: a.WorkerID, PreviousAttemptID: prior, AttemptID: b.Attempt, WorkloadGeneration: b.Generation, Allocation: x.s.Plan.Slots[0].Allocation, NodeAuthorizationHash: strings.Repeat("b", 64), PrivateHash: start.PrivateHash, PrecisionHash: a.PrecisionHash}, relayRecovery{}))
	if _, err := old.Poll(ctx); err == nil {
		t.Fatal("replaced attempt polled")
	}
	if err := old.Start(ctx, start); err == nil {
		t.Fatal("replaced attempt recorded start")
	}
	o := x.options[0]
	o.Workload = b
	p, err := managedtls.New(x.issue(t, b), x.f.verify(trust.Coordinator, x.f.serverID), x.f.clock)
	check(t, err)
	o.Policy = func() (*managedtls.Policy, error) { return p, nil }
	current, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	r, err := current.Poll(ctx)
	check(t, err)
	if r.Assignment == nil || *r.Assignment != a {
		t.Fatal("retry changed replay assignment")
	}
	check(t, current.Start(ctx, start))
}

func TestLearnerRelayPinsServerBeforeRequest(t *testing.T) {
	x := newLearnerRelay(t)
	f := x.f
	var calls atomic.Int32
	serverPolicy, err := learnerapi.NewPolicy(f.leaf(t, trust.Coordinator, id(t)), f.a, artifactTrust{f}, f.clock)
	check(t, err)
	server := serveCapabilities(t, serverPolicy, http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
	// Deliberately broad identity verifier: LearnerClient must narrow it before
	// sending any data, not merely inspect the response after a mutation.
	p, err := managedtls.New(x.identities[0], func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}, f.clock)
	check(t, err)
	o := x.options[0]
	o.Endpoint = server.URL
	o.Policy = func() (*managedtls.Policy, error) { return p, nil }
	c, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	if _, err = c.Poll(context.Background()); err == nil || calls.Load() != 0 {
		t.Fatal("foreign coordinator received request")
	}
}

func TestLearnerRelayReauthenticatesKeepalive(t *testing.T) {
	for _, failure := range []string{"revoked", "stale"} {
		t.Run(failure, func(t *testing.T) {
			x := newLearnerRelay(t)
			p, err := x.options[0].Policy()
			check(t, err)
			client := clientFor(t, p.ClientConfig())
			data, err := json.Marshal(x.pollRequest())
			check(t, err)
			post := func() (int, bool) {
				reused := false
				r, err := http.NewRequest(http.MethodPost, x.url+learnerapi.Route, bytes.NewReader(data))
				check(t, err)
				r = r.WithContext(httptrace.WithClientTrace(r.Context(), &httptrace.ClientTrace{GotConn: func(c httptrace.GotConnInfo) { reused = c.Reused }}))
				response, err := client.Do(r)
				check(t, err)
				_, err = io.Copy(io.Discard, response.Body)
				check(t, err)
				check(t, response.Body.Close())
				return response.StatusCode, reused
			}
			if code, _ := post(); code != 200 {
				t.Fatal(code)
			}
			x.f.mu.Lock()
			if failure == "stale" {
				x.f.now = x.f.now.Add(trust.SnapshotLifetime + time.Second)
			} else {
				x.f.snapshot.Payload.Generation++
				x.f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: x.bindings[0].Principal, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
				x.f.signSnapshot(t)
			}
			x.f.mu.Unlock()
			if code, reused := post(); code != 401 || !reused {
				t.Fatal("keepalive bypassed current trust", code, reused)
			}
		})
	}
}
