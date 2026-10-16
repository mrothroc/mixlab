package integration

import (
	"bytes"
	"context"
	"crypto"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerprobe"
)

type learnerHTTPFixture struct {
	f               *tlsFixture
	store           *nodeagent.Store
	prepare         clusterapp.NodeLearnerPrepareRequest
	coordinator     managedtls.Identity
	policy          *managedtls.Policy
	url             string
	calls           atomic.Int32
	initializations atomic.Int32
	capabilities    nodeagent.Capabilities
	requirements    recruitment.Requirements
}

type learnerHTTPJobs struct {
	*clusterapp.NodeJobs
	calls *atomic.Int32
}

func (j learnerHTTPJobs) PrepareLearner(ctx context.Context, p trust.AuthenticatedPrincipal, q clusterapp.NodeLearnerPrepareRequest) (clusterapp.NodeLearnerPrepared, error) {
	j.calls.Add(1)
	return j.NodeJobs.PrepareLearner(ctx, p, q)
}
func (j learnerHTTPJobs) CredentialLearner(ctx context.Context, p trust.AuthenticatedPrincipal, id string, q clusterapp.NodeLearnerCredentialRequest) error {
	j.calls.Add(1)
	return j.NodeJobs.CredentialLearner(ctx, p, id, q)
}
func (j learnerHTTPJobs) StartLearner(ctx context.Context, p trust.AuthenticatedPrincipal, id string, q clusterapp.NodeLearnerStartRequest) error {
	j.calls.Add(1)
	return j.NodeJobs.StartLearner(ctx, p, id, q)
}
func (j learnerHTTPJobs) FinalizeLearner(ctx context.Context, p trust.AuthenticatedPrincipal, id string, q authenticated.Commit) error {
	j.calls.Add(1)
	return j.NodeJobs.FinalizeLearner(ctx, p, id, q)
}

func learnerHTTPPath(t *testing.T, kind statehome.Kind) statehome.Path {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: kind})
	check(t, err)
	return p
}

func newLearnerHTTP(t *testing.T) *learnerHTTPFixture {
	return newLearnerHTTPWithPlan(t, nil)
}

func newLearnerHTTPWithPlan(t *testing.T, configure func(*run.Plan)) *learnerHTTPFixture {
	return newLearnerHTTPWithBuild(t, configure, strings.Repeat("a", 64))
}

func newLearnerHTTPWithBuild(t *testing.T, configure func(*run.Plan), build string) *learnerHTTPFixture {
	t.Helper()
	f := newTLSFixture(t)
	s, original, _, requirements := capabilityNodeWithBuild(t, f, t.TempDir(), build)
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Worker)
	f.signSnapshot(t)
	x := &learnerHTTPFixture{f: f, store: s, coordinator: f.leaf(t, trust.Coordinator, id(t))}
	ctx := context.Background()
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	caps, err := s.Capabilities(ctx, actor, f.now)
	check(t, err)
	x.capabilities, x.requirements = caps, requirements
	lease, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	var plan run.Plan
	b, err := os.ReadFile("../../diloco/testdata/plan.json")
	check(t, err)
	check(t, json.Unmarshal(b, &plan))
	member := distributed.LearnerMember{WorkerID: id(t), ParticipantID: caps.Node, IslandID: "metal", DatasetPartitionID: "partition0"}
	plan.Cohort, err = distributed.NewLearnerCohort(lease.Run, 1, []distributed.LearnerMember{member})
	check(t, err)
	allocation := run.Allocation{Provider: "node_agent", NodeID: caps.Node, LeaseID: lease.ID, JobID: id(t)}
	allocation.RefHash, err = run.Digest(allocation)
	check(t, err)
	capHash, err := run.Digest(caps)
	check(t, err)
	plan.Slots = []run.Slot{{Member: member, CapabilityHash: capHash, Allocation: allocation}}
	coordinator, err := trust.AuthenticatePrincipal(f.a, f.view, x.coordinator.Chain, f.now)
	check(t, err)
	plan.Coordinator = run.Coordinator{ClusterID: f.a.Cluster(), PrincipalID: coordinator.Principal, Role: "coordinator", Audience: "learner-run"}
	plan.CreatedAt, plan.ExpiresAt = f.now.Unix(), f.now.Add(time.Minute).Unix()
	config := json.RawMessage(fmt.Sprintf(`{"model_dim":16,"clock":%d}`, f.now.Unix()))
	plan.Contract.BuildHash = build
	plan.Contract.ConfigHash = nodejob.Hash(config)
	if configure != nil {
		configure(&plan)
	}
	plan.SHA256, err = plan.Digest()
	check(t, err)
	m := nodejob.LearnerManifest{Version: nodejob.LearnerVersion, Controller: actor.Principal, Attempt: id(t), WorkloadGeneration: 1, Plan: plan, Slot: plan.Slots[0], Config: config, DatasetSelector: "toy", Limits: requirements.Limits}
	q, err := m.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	x.prepare = clusterapp.NodeLearnerPrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: nodejob.SignedLearner{Manifest: m, Proof: proof}}
	cert := original.ServerConfig().Certificates[0]
	nodeIdentity := managedtls.Identity{Chain: cert.Certificate, Key: cert.PrivateKey.(crypto.Signer)}
	policy, err := managedtls.New(nodeIdentity, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}, f.clock)
	check(t, err)
	root := learnerHTTPPath(t, statehome.Agent)
	x.policy = policy
	credentialPath := func(scope workload.Scope) (statehome.Path, error) {
		return statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root.Dir(), "job-"+scope.Job+"-"+scope.Attempt)}, statehome.Context{Kind: statehome.Agent})
	}
	wp := nodeagent.WorkloadPorts{Clock: f.clock, Initialize: func(ctx context.Context, scope workload.Scope) error {
		x.initializations.Add(1)
		p, err := credentialPath(scope)
		if err != nil {
			return err
		}
		c, err := nodecredentials.InitializeTransport(ctx, p, f.a, scope, "file")
		if err != nil {
			return err
		}
		return c.Close()
	}, Request: func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
		p, err := credentialPath(scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		c, err := nodecredentials.OpenTransport(p, f.a, scope)
		if err != nil {
			return workload.SignedRequest{}, err
		}
		defer func() { _ = c.Close() }()
		return c.PrepareRequest(ctx, func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
			return trust.SignPrincipalProof(f.a, nodeIdentity.Chain, nodeIdentity.Key, f.view, q, f.now)
		}, f.view, f.now)
	}}
	c := plan.Contract
	probe := workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: c.BuildHash, ConfigHash: c.ConfigHash, ProgramHash: c.ProgramHash, WeightLayoutHash: c.WeightLayoutHash, OptimizerHash: c.InnerOptimizerHash, DType: "fp32", Learner: &workerprobe.LearnerPlan{Outer: plan.Outer, BatchTokens: 16, SequenceLength: 4, Precision: &precision.Policy{ComputeDType: "float32", Backend: "metal"}}}
	jobs, err := clusterapp.NewNodeJobs(ctx, clusterapp.NodeJobOptions{Store: s, Node: caps.Node, Anchor: f.a, CredentialRoot: root, Workload: wp, RelayAddress: "127.0.0.1:32499", Clock: f.clock, CurrentTrust: func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		return f.a, f.view, nil
	}, SignRecovery: func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(f.a, nodeIdentity.Chain, nodeIdentity.Key, f.view, q, f.now)
	}, Preparation: nodeagent.PreparationPorts{Clock: f.clock, Probe: func(context.Context) (workerprobe.Report, error) { return caps.Probe, nil }, DatasetID: func(_ context.Context, path string) (string, error) {
		if path != "/private/local/train.bin" {
			return "", fmt.Errorf("nonlocal dataset path")
		}
		return requirements.DatasetID, nil
	}, ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error { return fmt.Errorf("unexpected artifact") }}, LearnerProbe: func(context.Context, []byte) (workerprobe.Plan, error) { return probe, nil }})
	check(t, err)
	h, err := clusterapp.NodeManagementHandler(s, policy, f.clock, learnerHTTPJobs{jobs, &x.calls})
	check(t, err)
	x.url = serveCapabilities(t, policy, h).URL
	return x
}

func (x *learnerHTTPFixture) client(t *testing.T, identity managedtls.Identity) (*clusterapp.NodeClient, *http.Client) {
	t.Helper()
	p, err := managedtls.New(identity, x.f.verify(trust.Node, x.prepare.Signed.Manifest.Slot.Allocation.NodeID), x.f.clock)
	check(t, err)
	c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(x.url, "https://"), x.prepare.Signed.Manifest.Slot.Allocation.NodeID, func() (*managedtls.Policy, error) { return p, nil })
	check(t, err)
	h, err := p.HTTPClient(5 * time.Second)
	check(t, err)
	t.Cleanup(h.CloseIdleConnections)
	return c, h
}

func learnerHTTPBody(t *testing.T, value any) []byte {
	t.Helper()
	b, err := json.Marshal(value)
	check(t, err)
	return b
}

func learnerHTTPRequest(t *testing.T, client *http.Client, method, url, contentType string, body []byte) (int, []byte) {
	t.Helper()
	r, err := http.NewRequest(method, url, bytes.NewReader(body))
	check(t, err)
	r.Header.Set("Content-Type", contentType)
	resp, err := client.Do(r)
	check(t, err)
	defer func() { _ = resp.Body.Close() }()
	b, err := io.ReadAll(resp.Body)
	check(t, err)
	return resp.StatusCode, b
}

func TestLearnerHTTPClosedSchemasRolesAndMethods(t *testing.T) {
	x := newLearnerHTTP(t)
	_, controller := x.client(t, x.f.client)
	_, coordinator := x.client(t, x.coordinator)
	_, authority := x.client(t, x.f.server)
	job := x.prepare.Signed.Manifest.Slot.Allocation.JobID
	for _, op := range []struct {
		name, method string
		body         any
		limit        int
	}{
		{"prepare", http.MethodPut, x.prepare, 2 << 20},
		{"credential", http.MethodPut, clusterapp.NodeLearnerCredentialRequest{}, 1 << 20},
		{"start", http.MethodPost, clusterapp.NodeLearnerStartRequest{Key: id(t)}, 4096},
		{"finalize", http.MethodPut, authenticated.Commit{}, 2 << 20},
	} {
		t.Run(op.name, func(t *testing.T) {
			url := x.url + "/v1/agent/learners/" + job + "/" + op.name
			b := learnerHTTPBody(t, op.body)
			var fields map[string]json.RawMessage
			check(t, json.Unmarshal(b, &fields))
			var duplicate []byte
			for name, value := range fields {
				duplicate = append([]byte(fmt.Sprintf("{%q:%s,", name, value)), b[1:]...)
				break
			}
			for _, tc := range []struct {
				name        string
				body        []byte
				contentType string
			}{
				{"unknown", append(bytes.Clone(b[:len(b)-1]), []byte(`,"assignment":{}}`)...), "application/json"},
				{"local_path", append(bytes.Clone(b[:len(b)-1]), []byte(`,"private_path":"/tmp/remote"}`)...), "application/json"},
				{"trust_override", append(bytes.Clone(b[:len(b)-1]), []byte(`,"snapshot":{}}`)...), "application/json"},
				{"trailing_object", append(bytes.Clone(b), []byte(`{}`)...), "application/json"},
				{"duplicate_field", duplicate, "application/json"},
				{"null", []byte(`null`), "application/json"},
				{"oversize", bytes.Repeat([]byte(" "), op.limit+1), "application/json"},
				{"media_type", b, "text/plain"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					before := x.calls.Load()
					code, _ := learnerHTTPRequest(t, controller, op.method, url, tc.contentType, tc.body)
					if code != http.StatusBadRequest || x.calls.Load() != before {
						t.Fatalf("status=%d dispatched=%d", code, x.calls.Load()-before)
					}
				})
			}
			for name, value := range fields {
				if len(value) == 0 || value[0] != '{' {
					continue
				}
				extra := bytes.Replace(b, value, append([]byte(`{"train_pattern":"/tmp/remote",`), value[1:]...), 1)
				before := x.calls.Load()
				code, _ := learnerHTTPRequest(t, controller, op.method, url, "application/json", extra)
				if code != http.StatusBadRequest || x.calls.Load() != before {
					t.Fatalf("nested %s: status=%d", name, code)
				}
			}
			for _, method := range []string{http.MethodGet, http.MethodDelete, http.MethodPatch, http.MethodPost, http.MethodPut} {
				if method == op.method {
					continue
				}
				before := x.calls.Load()
				code, _ := learnerHTTPRequest(t, controller, method, url, "application/json", b)
				if code != http.StatusNotFound || x.calls.Load() != before {
					t.Fatalf("method %s: status=%d", method, code)
				}
			}
			for _, client := range []*http.Client{authority, coordinator} {
				if client == coordinator && op.name == "finalize" {
					continue
				}
				before := x.calls.Load()
				code, _ := learnerHTTPRequest(t, client, op.method, url, "application/json", b)
				if code != http.StatusForbidden || x.calls.Load() != before {
					t.Fatalf("role: status=%d", code)
				}
			}
			for _, suffix := range []string{"?scope=other", "/extra"} {
				before := x.calls.Load()
				code, _ := learnerHTTPRequest(t, controller, op.method, url+suffix, "application/json", b)
				if code != http.StatusNotFound || x.calls.Load() != before {
					t.Fatalf("route: status=%d", code)
				}
			}
		})
	}
	for _, q := range []clusterapp.NodeLearnerPrepareRequest{{Signed: x.prepare.Signed}, {ExpectedLeaseVersion: 1}} {
		code, _ := learnerHTTPRequest(t, controller, http.MethodPut, x.url+"/v1/agent/learners/"+job+"/prepare", "application/json", learnerHTTPBody(t, q))
		if code != http.StatusBadRequest {
			t.Fatal(code)
		}
	}
	for _, key := range []string{"", "../other", strings.Repeat("A", 32)} {
		code, _ := learnerHTTPRequest(t, controller, http.MethodPost, x.url+"/v1/agent/learners/"+job+"/start", "application/json", learnerHTTPBody(t, clusterapp.NodeLearnerStartRequest{Key: key}))
		if code != http.StatusBadRequest {
			t.Fatal(code)
		}
	}
	if x.calls.Load() != 0 {
		t.Fatal("rejected request reached node application")
	}
}

func TestLearnerHTTPReplayAndScope(t *testing.T) {
	x := newLearnerHTTP(t)
	ctx := context.Background()
	client, raw := x.client(t, x.f.client)
	coordinator, _ := x.client(t, x.coordinator)
	foreignController, _ := x.client(t, x.f.leaf(t, trust.Controller, id(t)))
	foreignCoordinator, _ := x.client(t, x.f.leaf(t, trust.Coordinator, id(t)))
	job := x.prepare.Signed.Manifest.Slot.Allocation.JobID
	if _, err := foreignController.PrepareLearner(ctx, x.prepare); err == nil {
		t.Fatal("foreign controller prepared job")
	}
	prepared, err := client.PrepareLearner(ctx, x.prepare)
	check(t, err)
	again, err := client.PrepareLearner(ctx, x.prepare)
	check(t, err)
	if !reflect.DeepEqual(prepared, again) || x.initializations.Load() != 1 {
		t.Fatal("prepare replay changed registration or key")
	}
	for _, change := range []func(*nodejob.LearnerManifest){
		func(m *nodejob.LearnerManifest) { m.Attempt = id(t) },
		func(m *nodejob.LearnerManifest) { m.Slot.Member.WorkerID = id(t) },
		func(m *nodejob.LearnerManifest) { m.Plan.Coordinator.Audience = "another" },
	} {
		bad := x.prepare
		change(&bad.Signed.Manifest)
		if _, err := client.PrepareLearner(ctx, bad); err == nil {
			t.Fatal("changed preparation scope accepted")
		}
	}
	start := clusterapp.NodeLearnerStartRequest{Key: id(t)}
	if err := client.StartLearner(ctx, job, start); err == nil {
		t.Fatal("started without credential")
	}
	grantQ, err := prepared.Request.GrantRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, grantQ, x.f.now)
	check(t, err)
	grant := workload.Grant{Request: prepared.Request, Proof: proof}
	issuer, err := workload.Initialize(ctx, learnerHTTPPath(t, statehome.Authority), workload.Authority{Anchor: x.f.a, Issuer: x.f.issuer, Key: x.f.issuerKey}, x.f.view, x.f.now)
	check(t, err)
	result, err := issuer.Issue(ctx, x.f.client.Chain, grant, x.f.view, x.f.now)
	check(t, err)
	credential := clusterapp.NodeLearnerCredentialRequest{Grant: grant, Credential: result}
	for _, field := range []string{"worker", "run", "attempt", "coordinator"} {
		bad := credential
		switch field {
		case "worker":
			bad.Grant.Request.Request.Scope.Worker = id(t)
		case "run":
			bad.Grant.Request.Request.Scope.Run = id(t)
		case "attempt":
			bad.Grant.Request.Request.Scope.Attempt = id(t)
		case "coordinator":
			bad.Grant.Request.Request.Scope.Coordinator = id(t)
		}
		if err := client.CredentialLearner(ctx, job, bad); err == nil {
			t.Fatal("credential scope accepted:", field)
		}
	}
	if err := foreignController.CredentialLearner(ctx, job, credential); err == nil {
		t.Fatal("foreign controller installed credential")
	}
	if err := client.CredentialLearner(ctx, id(t), credential); err == nil {
		t.Fatal("credential route substituted")
	}
	check(t, client.CredentialLearner(ctx, job, credential))
	before, err := x.store.ActiveExecution(ctx)
	check(t, err)
	check(t, client.CredentialLearner(ctx, job, credential))
	after, err := x.store.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) {
		t.Fatal("credential replay changed durable execution")
	}
	if err := foreignController.StartLearner(ctx, job, start); err == nil {
		t.Fatal("foreign controller started job")
	}
	check(t, client.StartLearner(ctx, job, start))
	before, err = x.store.ActiveExecution(ctx)
	check(t, err)
	check(t, client.StartLearner(ctx, job, start))
	after, err = x.store.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) || after.Learner.Assignment.TrainPattern != "/private/local/train.bin" {
		t.Fatal("start replay changed local assignment")
	}
	if err := client.StartLearner(ctx, job, clusterapp.NodeLearnerStartRequest{Key: id(t)}); err == nil {
		t.Fatal("start replay changed key")
	}
	p := x.prepare.Signed.Manifest.Plan
	commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: p.Cohort.RunID, PlanHash: p.SHA256, Generation: p.Cohort.Generation, CohortHash: p.Cohort.CohortHash}, Coordinator: p.Coordinator, Registrations: []run.Registration{prepared.Registration}, CommittedAt: x.f.now.Unix()}
	commit.Binding.CommitHash, err = commit.Digest()
	check(t, err)
	proof, err = trust.SignPrincipalProof(x.f.a, x.coordinator.Chain, x.coordinator.Key, x.f.view, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.RunCommit, Digest: commit.Binding.CommitHash, Context: p.Cohort.RunID, Audience: p.Coordinator.Audience}, x.f.now)
	check(t, err)
	finalize := authenticated.Commit{Body: commit, Proof: proof}
	for _, c := range []*clusterapp.NodeClient{foreignController, foreignCoordinator} {
		if err := c.FinalizeLearner(ctx, job, finalize); err == nil {
			t.Fatal("foreign principal finalized learner")
		}
	}
	bad := finalize
	bad.Body.Binding.RunID = id(t)
	if err := coordinator.FinalizeLearner(ctx, job, bad); err == nil {
		t.Fatal("commit run substituted")
	}
	check(t, client.FinalizeLearner(ctx, job, finalize))
	before, err = x.store.ActiveExecution(ctx)
	check(t, err)
	check(t, coordinator.FinalizeLearner(ctx, job, finalize))
	after, err = x.store.ActiveExecution(ctx)
	check(t, err)
	if !reflect.DeepEqual(before, after) || after.Lease.State != nodeagent.Running || after.Lease.Coordinator != p.Coordinator.PrincipalID {
		t.Fatal("finalize replay changed handoff")
	}
	code, body := learnerHTTPRequest(t, raw, http.MethodPut, x.url+"/v1/agent/learners/"+id(t)+"/finalize", "application/json", learnerHTTPBody(t, finalize))
	if code != http.StatusConflict || strings.Contains(string(body), "/private/") || strings.Contains(string(body), "no such file") {
		t.Fatalf("unsafe error: %d %s", code, body)
	}
}

func TestLearnerHTTPClientRejectsResponseSubstitution(t *testing.T) {
	x := newLearnerHTTP(t)
	client, _ := x.client(t, x.f.client)
	prepared, err := client.PrepareLearner(context.Background(), x.prepare)
	check(t, err)
	serverPolicy := x.policy
	clientPolicy, err := managedtls.New(x.f.client, x.f.verify(trust.Node, x.prepare.Signed.Manifest.Slot.Allocation.NodeID), x.f.clock)
	check(t, err)
	// This server has valid TLS but returns a response for a different job or
	// workload. Pinning the peer does not replace application binding checks.
	for _, field := range []string{"valid", "registration_attempt", "registration_worker", "workload_run", "workload_attempt", "ack_job"} {
		t.Run(field, func(t *testing.T) {
			bad := prepared
			switch field {
			case "registration_attempt":
				bad.Registration.AttemptID = id(t)
			case "registration_worker":
				bad.Registration.Slot.Member.WorkerID = id(t)
			case "workload_run":
				bad.Request.Request.Scope.Run = id(t)
			case "workload_attempt":
				bad.Request.Request.Scope.Attempt = id(t)
			}
			var calls atomic.Int32
			h := serverPolicy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				if field == "ack_job" {
					_ = json.NewEncoder(w).Encode(clusterapp.NodeLearnerAck{Job: id(t)})
					return
				}
				_ = json.NewEncoder(w).Encode(bad)
			}))
			server := serveCapabilities(t, serverPolicy, h)
			c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(server.URL, "https://"), x.prepare.Signed.Manifest.Slot.Allocation.NodeID, func() (*managedtls.Policy, error) { return clientPolicy, nil })
			check(t, err)
			if field == "ack_job" {
				job := x.prepare.Signed.Manifest.Slot.Allocation.JobID
				for _, command := range []func() error{
					func() error {
						return c.CredentialLearner(context.Background(), job, clusterapp.NodeLearnerCredentialRequest{})
					},
					func() error {
						return c.StartLearner(context.Background(), job, clusterapp.NodeLearnerStartRequest{Key: id(t)})
					},
					func() error { return c.FinalizeLearner(context.Background(), job, authenticated.Commit{}) },
				} {
					if err := command(); err == nil {
						t.Fatal("substituted acknowledgment accepted")
					}
				}
			} else {
				_, err := c.PrepareLearner(context.Background(), x.prepare)
				if field == "valid" {
					check(t, err)
				} else if err == nil {
					t.Fatal("substituted preparation accepted")
				}
			}
			if calls.Load() == 0 {
				t.Fatal("request did not reach authenticated response fixture")
			}
		})
	}
}
