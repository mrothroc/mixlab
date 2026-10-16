//go:build darwin || linux

package integration

import (
	"context"
	"crypto"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type partialArtifactWire struct {
	armed atomic.Bool
	reads atomic.Int32
	cut   chan int
}

// Run the real grant, authentication, ledger and store handlers before cutting
// the wire response. The declared length stays intact, so the real client sees
// an authenticated HTTP 200 followed by a truncated body, not a fake error.
func (p *partialArtifactWire) wrap(t *testing.T, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != artifactapi.ReadRoute {
			h.ServeHTTP(w, r)
			return
		}
		p.reads.Add(1)
		if !p.armed.CompareAndSwap(true, false) {
			h.ServeHTTP(w, r)
			return
		}
		recorder := httptest.NewRecorder()
		h.ServeHTTP(capturedLearnerResponse{recorder, w}, r)
		if recorder.Code != http.StatusOK || recorder.Body.Len() < 2 {
			t.Errorf("artifact was not authorized before interruption: HTTP %d", recorder.Code)
			http.Error(w, "unexpected artifact rejection", http.StatusInternalServerError)
			return
		}
		for key, values := range recorder.Header() {
			w.Header()[key] = append([]string(nil), values...)
		}
		w.WriteHeader(http.StatusOK)
		n, err := w.Write(recorder.Body.Bytes()[:recorder.Body.Len()/2])
		if err != nil {
			t.Error(err)
		}
		w.(http.Flusher).Flush()
		dropLearnerResponse(t, w)
		p.cut <- n
	})
}

func partialArtifactHandler(t *testing.T, f *recoveryFixture, grantsPath, ledgerPath statehome.Path) (*managedtls.Policy, http.Handler) {
	t.Helper()
	x := f.x
	policy, err := learnerapi.NewPolicy(x.coordinator, x.f.a, artifactTrust{x.f}, x.f.clock)
	check(t, err)
	grants, err := learnerapi.OpenGrants(grantsPath)
	check(t, err)
	ledger, err := artifactapi.OpenLedger(ledgerPath)
	check(t, err)
	semantic, err := (learnerapi.Gateway{Grants: grants, Policy: policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: f.state(t).Plan.Coordinator, Model: f.model, Clock: x.f.clock, Timeout: 10 * time.Second, Sign: func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(x.f.a, x.coordinator.Chain, x.coordinator.Key, x.f.view, q, x.f.now)
	}}).Handler()
	check(t, err)
	artifacts, err := (artifactapi.Gateway{Policy: policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: f.state(t).Plan.Coordinator, Repository: f.model.Repository, Store: f.model.Artifacts.(coordinatorlocal.Artifacts).Store, Ledger: ledger, Clock: x.f.clock, TransferTimeout: 10 * time.Second}).Handler()
	check(t, err)
	mux := http.NewServeMux()
	mux.Handle(learnerapi.Route, semantic)
	mux.Handle("/v1/diloco/artifacts/", artifacts)
	return policy, mux
}

// The credential test port issues the exact active binding under the fixture
// authority. Production workload/node TLS policies and HTTP handlers still
// verify it; no credential or network authority crosses learner IPC.
func partialArtifactClient(t *testing.T, f *recoveryFixture, x nodeagent.LocalExecution, endpoint string) *clusterapp.LearnerClient {
	t.Helper()
	b := x.Learner.Credential.Binding
	k := key(t)
	tls := f.x.f
	der, err := certificates.IssueWorkload(tls.a, tls.issuer, tls.issuerKey, k.Public(), b, time.Unix(b.ExpiresAt, 0), tls.now)
	check(t, err)
	verify := tls.verify(trust.Coordinator, f.state(t).Plan.Coordinator.PrincipalID)
	workloadPolicy, err := managedtls.New(managedtls.Identity{Chain: [][]byte{der, tls.issuer, tls.a.DER()}, Key: k}, verify, tls.clock)
	check(t, err)
	cert := f.x.policy.ServerConfig().Certificates[0]
	nodePolicy, err := managedtls.New(managedtls.Identity{Chain: cert.Certificate, Key: cert.PrivateKey.(crypto.Signer)}, verify, tls.clock)
	check(t, err)
	c, err := clusterapp.NewLearnerClientWithPolicy(clusterapp.LearnerClientOptions{Plan: x.Learner.Signed.Manifest.Plan, Commit: *x.Learner.Commit, Workload: b, Endpoint: endpoint, Anchor: tls.a, Trust: artifactTrust{tls}, Clock: tls.clock, Policy: func() (*managedtls.Policy, error) { return workloadPolicy, nil }, NodePolicy: func() (*managedtls.Policy, error) { return nodePolicy, nil }})
	check(t, err)
	return c
}

type partialArtifactPeer struct {
	*clusterapp.LearnerClient
	f      *recoveryFixture
	errors chan error
}

// Deterministically drive the same production tick used by the service, without
// a concurrent scheduler or installed-service lifecycle in this fixture.
func (p *partialArtifactPeer) Poll(ctx context.Context) (private.RelayResponse, error) {
	if err := p.f.model.Tick(ctx, p.f.assignment.Binding.RunID, 200); err != nil {
		return private.RelayResponse{}, err
	}
	return p.LearnerClient.Poll(ctx)
}
func (p *partialArtifactPeer) Read(ctx context.Context, a run.Assignment, b run.Blob, w io.Writer) error {
	err := p.LearnerClient.Read(ctx, a, b, w)
	if err != nil {
		p.errors <- err
	}
	return err
}

type partialArtifactCheckpoint struct {
	*clusterapp.LearnerClient
	stop error
}

func (p partialArtifactCheckpoint) Start(ctx context.Context, s private.Start) error {
	if err := p.LearnerClient.Start(ctx, s); err != nil {
		return err
	}
	return p.stop
}

// Publish a real, acknowledged start through Runtime and HTTP, then stop before
// Train. Recovery must have private history; a failed very first model download
// must not be allowed to initialize new RNG state under a replacement attempt.
func primePartialArtifactCheckpoint(t *testing.T, f *recoveryFixture, path statehome.Path, c *clusterapp.LearnerClient) []byte {
	t.Helper()
	a := f.assignment
	x, err := f.x.store.ActiveExecution(context.Background())
	check(t, err)
	hosted := x.Learner.Assignment
	store, err := learner.Open(path, hosted.PrivateMaxBytes)
	check(t, err)
	stop := errors.New("fixture checkpoint established")
	calls := 0
	r := learner.Runtime{Store: store, Peer: partialArtifactCheckpoint{c, stop}, Now: func() int64 { return f.x.f.now.Unix() }, Scope: learner.Scope{RunID: a.Binding.RunID, PlanHash: a.Binding.PlanHash, CohortHash: a.Binding.CohortHash, Generation: a.Binding.Generation, WorkerID: a.WorkerID, PartitionID: a.PartitionID, ParticipantID: hosted.Member.ParticipantID, IslandID: hosted.Member.IslandID, Contract: a.Contract, Coordinator: a.Coordinator}, Factory: func(model []optimizer.Tensor) (learner.Engine, error) {
		return &relayEngine{calls: &calls, state: learner.Snapshot{Model: relayClone(model), Optimizer: []byte("moment-0"), Sampler: []byte("cursor-0"), Schedule: []byte("schedule"), Metadata: []byte("metadata"), Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: a.Binding.RunID, WorkerID: a.WorkerID, PartitionID: a.PartitionID, Seed: 99}, Precision: precision.Policy{ComputeDType: "float32", Backend: "metal"}}}, nil
	}}
	_, err = r.Execute(context.Background(), a)
	if !errors.Is(err, stop) || calls != 0 {
		t.Fatalf("checkpoint setup trained or failed: calls=%d err=%v", calls, err)
	}
	b, err := path.ReadFileLimit("head.json", 4<<20)
	check(t, err)
	return b
}
