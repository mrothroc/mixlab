package integration

import (
	"bytes"
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type artifactTrust struct{ f *tlsFixture }

func (a artifactTrust) Current(_ context.Context, _ time.Time) (trust.VerifiedSnapshot, error) {
	a.f.mu.RLock()
	defer a.f.mu.RUnlock()
	return a.f.view, nil
}

type artifactFixture struct {
	f         *tlsFixture
	g         artifactapi.Gateway
	s         coordinator.State
	clients   []managedtls.Identity
	clientIDs []string
	storePath statehome.Path
}

func artifactPath(t *testing.T) statehome.Path {
	t.Helper()
	dir, e := filepath.EvalSymlinks(t.TempDir())
	check(t, e)
	check(t, os.Chmod(dir, 0700))
	p, e := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal, Role: "coordinator"})
	check(t, e)
	return p
}
func newArtifactFixture(t *testing.T, ref artifact.Ref) *artifactFixture {
	t.Helper()
	f := newTLSFixture(t)
	f.server = f.leaf(t, trust.Coordinator, f.serverID)
	a := &artifactFixture{f: f}
	f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Node)
	f.signSnapshot(t)
	for range 2 {
		id := id(t)
		key := key(t)
		envelope, e := ecdh.X25519().GenerateKey(rand.Reader)
		check(t, e)
		der, e := certificates.IssueNode(f.a, f.issuer, f.issuerKey, id, key.Public(), envelope.PublicKey().Bytes(), f.now)
		check(t, e)
		a.clients = append(a.clients, managedtls.Identity{Chain: [][]byte{der, f.issuer, f.a.DER()}, Key: key})
		a.clientIDs = append(a.clientIDs, id)
	}
	policy, e := managedtls.New(f.server, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return trust.AuthenticatePrincipal(f.a, f.view, chain, now)
	}, f.clock)
	check(t, e)
	repo, e := coordinatorlocal.Open(artifactPath(t))
	check(t, e)
	a.storePath = artifactPath(t)
	store, e := artifactlocal.Open(a.storePath)
	check(t, e)
	ledger, e := artifactapi.OpenLedger(artifactPath(t))
	check(t, e)
	h := strings.Repeat("a", 64)
	c := run.Coordinator{ClusterID: f.a.Cluster(), PrincipalID: f.serverID, Role: "coordinator", Audience: "artifact-test"}
	members := []distributed.LearnerMember{}
	for i, id := range a.clientIDs {
		members = append(members, distributed.LearnerMember{WorkerID: fmt.Sprint("w", i), ParticipantID: id, IslandID: fmt.Sprint("mac", i), DatasetPartitionID: fmt.Sprint("p", i)})
	}
	cohort, e := distributed.NewLearnerCohort("run-test", 0, members)
	check(t, e)
	p := run.Plan{Format: run.PlanVersion, Cohort: cohort, Coordinator: c, Contract: run.Contract{ConfigHash: h, ProgramHash: h, WeightLayoutHash: h, InnerOptimizerHash: h, DatasetHash: h, BuildHash: h}, BaseModel: run.Blob{Ref: ref, Kind: "model_f32"}, Outer: run.Outer{Kind: "nesterov", LR: 1, Aggregation: "uniform", InnerSteps: 2, TotalSteps: 4, Accumulation: 1}, ArtifactStoreID: "store", Endpoint: "https://test.invalid", CreatedAt: f.now.Unix() - 30, ExpiresAt: f.now.Unix() + 3600}
	for i, m := range members {
		allocation := run.Allocation{Provider: "node_agent", NodeID: m.ParticipantID, LeaseID: fmt.Sprint("lease", i), JobID: fmt.Sprint("job", i)}
		allocation.RefHash, e = run.Digest(allocation)
		check(t, e)
		p.Slots = append(p.Slots, run.Slot{Member: m, CapabilityHash: h, Allocation: allocation})
	}
	p.SHA256, e = p.Digest()
	check(t, e)
	check(t, p.Validate())
	commit := run.Commit{Format: run.CommitVersion, Binding: run.Binding{RunID: cohort.RunID, PlanHash: p.SHA256, CohortHash: cohort.CohortHash}, Coordinator: c, CommittedAt: f.now.Unix()}
	precision := precision.Policy{ComputeDType: "float32", Backend: "metal"}
	ph, e := precision.Digest()
	check(t, e)
	for _, slot := range p.Slots {
		commit.Registrations = append(commit.Registrations, run.Registration{Format: run.RegistrationVersion, RunID: cohort.RunID, PlanHash: p.SHA256, CohortHash: cohort.CohortHash, Slot: slot, Contract: p.Contract, BaseModelHash: ref.SHA256, PreparationHash: h, JobManifestHash: h, AttemptID: "attempt-0", Coordinator: c, Precision: precision, PrecisionHash: ph})
	}
	commit.Binding.CommitHash, e = commit.Digest()
	check(t, e)
	check(t, commit.ValidateFor(p))
	a.s = coordinator.State{Revision: 1, Phase: coordinator.Open, Plan: p, Commit: &commit, Policy: coordinator.EvaluationPolicy{WorkerID: members[0].WorkerID, ValidationHash: h, Split: "validation", BatchTokens: 8, SequenceLength: 4, Batches: 1, TimeoutSeconds: 60}, Base: p.BaseModel, Momentum: run.Blob{Ref: ref, Kind: "outer_state_f32"}, PrivateHashes: []string{"", ""}}
	for i, reg := range commit.Registrations {
		a.s.Assignments = append(a.s.Assignments, run.Assignment{Format: run.AssignmentVersion, ID: fmt.Sprint("assignment-", i), Binding: commit.Binding, Coordinator: c, WorkerID: reg.Slot.Member.WorkerID, BaseModel: p.BaseModel, OuterState: a.s.Momentum, Contract: p.Contract, PartitionID: reg.Slot.Member.DatasetPartitionID, InnerSteps: p.Outer.InnerSteps, PrecisionHash: ph, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Unix() + 600})
	}
	check(t, repo.Save(context.Background(), cohort.RunID, 0, coordinator.Change{Next: a.s}))
	a.g = artifactapi.Gateway{Policy: policy, Anchor: f.a, Trust: artifactTrust{f}, Coordinator: c, Repository: repo, Store: store, Ledger: ledger, Clock: f.clock, TransferTimeout: time.Minute}
	return a
}
func (a *artifactFixture) request(t *testing.T, index int, operation, id string) artifactapi.Request {
	t.Helper()
	assignment := a.s.Assignments[index]
	b := run.Grant{Format: "mixlab_artifact_access_grant_v1", ID: id, Binding: assignment.Binding, Coordinator: a.g.Coordinator, PrincipalID: a.clientIDs[index], ParticipantID: a.clientIDs[index], WorkerID: assignment.WorkerID, AssignmentID: assignment.ID, Operation: operation, Artifact: assignment.BaseModel, AccessClass: "global_base", IssuedAt: a.f.now.Unix(), ExpiresAt: a.f.now.Unix() + 300, MaximumUses: 8, Nonce: id}
	if operation == "create" {
		b.Artifact.Kind = "delta_f32"
		b.AccessClass = "outer_update"
		b.UploadID = id
	}
	return a.sign(t, b)
}
func (a *artifactFixture) sign(t *testing.T, b run.Grant) artifactapi.Request {
	t.Helper()
	var e error
	b.SHA256, e = b.Digest()
	check(t, e)
	check(t, b.Validate())
	proof, e := trust.SignPrincipalProof(a.f.a, a.f.server.Chain, a.f.server.Key, a.f.view, trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.ArtifactGrant, Digest: b.SHA256, Context: b.Binding.RunID, Audience: b.Coordinator.Audience}, a.f.now)
	check(t, e)
	return artifactapi.Request{Grant: authenticated.Grant{Body: b, Proof: proof}, Use: run.GrantUse{Format: "mixlab_artifact_use_v1", GrantHash: b.SHA256, GrantID: b.ID, Operation: b.Operation, UploadID: b.UploadID, IdempotencyID: "use-0"}}
}
func artifactRef(body []byte) artifact.Ref {
	h := sha256.Sum256(body)
	return artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(body))}
}

func TestArtifactGatewayTLSAndScope(t *testing.T) {
	body := []byte("synthetic artifact bytes")
	a := newArtifactFixture(t, artifactRef(body))
	handler, e := a.g.Handler()
	check(t, e)
	server := serveCapabilities(t, a.g.Policy, handler)
	clientPolicy, e := managedtls.New(a.clients[0], a.f.verify(trust.Coordinator, a.f.serverID), a.f.clock)
	check(t, e)
	client, e := clientPolicy.HTTPClient(time.Second * 10)
	check(t, e)
	defer client.CloseIdleConnections()
	send := func(q artifactapi.Request, route string, payload []byte) (int, []byte) {
		t.Helper()
		var b bytes.Buffer
		check(t, artifactapi.WriteRequest(&b, q))
		b.Write(payload)
		resp, e := client.Post(server.URL+route, "application/octet-stream", &b)
		check(t, e)
		defer func() { _ = resp.Body.Close() }()
		out, e := io.ReadAll(resp.Body)
		check(t, e)
		return resp.StatusCode, out
	}
	upload := a.request(t, 0, "create", "upload")
	if code, _ := send(upload, artifactapi.CreateRoute, body[:3]); code != 409 {
		t.Fatal(code)
	}
	for range 2 {
		if code, _ := send(upload, artifactapi.CreateRoute, body); code != 204 {
			t.Fatal(code)
		}
	}
	read := a.request(t, 0, "read", "read")
	if code, out := send(read, artifactapi.ReadRoute, nil); code != 200 || !bytes.Equal(out, body) {
		t.Fatal(code, string(out))
	}
	for name, change := range map[string]func(*run.Grant){
		"worker":      func(g *run.Grant) { g.WorkerID = "w1" },
		"participant": func(g *run.Grant) { g.ParticipantID = a.clientIDs[1] },
		"principal":   func(g *run.Grant) { g.PrincipalID = a.clientIDs[1] },
		"assignment":  func(g *run.Grant) { g.AssignmentID = "stale" },
		"artifact":    func(g *run.Grant) { g.Artifact.Ref.SHA256 = strings.Repeat("b", 64) },
		"expiry":      func(g *run.Grant) { g.ExpiresAt = g.IssuedAt + 1; g.IssuedAt -= 10; g.ExpiresAt -= 10 },
	} {
		t.Run(name, func(t *testing.T) {
			g := read.Grant.Body
			g.ID = name
			change(&g)
			q := a.sign(t, g)
			if code, _ := send(q, artifactapi.ReadRoute, nil); code != 403 {
				t.Fatal(code)
			}
		})
	}
	if code, _ := send(read, artifactapi.CreateRoute, body); code != 403 {
		t.Fatal("operation substitution", code)
	}
	tampered := read
	tampered.Grant.Proof.Signature = bytes.Clone(read.Grant.Proof.Signature)
	tampered.Grant.Proof.Signature[0] ^= 1
	if code, _ := send(tampered, artifactapi.ReadRoute, nil); code != 403 {
		t.Fatal("invalid grant signature", code)
	}
	wrongUse := read
	wrongUse.Use.GrantHash = strings.Repeat("b", 64)
	if code, _ := send(wrongUse, artifactapi.ReadRoute, nil); code != 403 {
		t.Fatal("use binding substitution", code)
	}
	// Even exact retries cannot trust a same-digest corrupt file.
	check(t, a.storePath.WriteFile(read.Grant.Body.Artifact.Ref.SHA256, bytes.Repeat([]byte("x"), len(body))))
	if code, _ := send(upload, artifactapi.CreateRoute, body); code != 409 {
		t.Fatal("corrupt stored blob accepted", code)
	}
	stale := a.s
	stale.Revision++
	stale.Phase = coordinator.Failed
	stale.Failure = "test terminal state"
	check(t, a.g.Repository.Save(context.Background(), stale.Plan.Cohort.RunID, a.s.Revision, coordinator.Change{Next: stale}))
	if code, _ := send(read, artifactapi.ReadRoute, nil); code != 403 {
		t.Fatal("terminal run still served artifacts", code)
	}
	a.f.mu.Lock()
	a.f.snapshot.Payload.Generation++
	a.f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: a.f.serverID, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	a.f.signSnapshot(t)
	a.f.mu.Unlock()
	if resp, e := client.Post(server.URL+artifactapi.ReadRoute, "application/octet-stream", bytes.NewReader(nil)); e == nil {
		_ = resp.Body.Close()
		t.Fatal("revoked coordinator accepted")
	}
}
