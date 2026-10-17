package integration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

func nodeRolloverView(t *testing.T, x *nodeJobsFixture, current trust.VerifiedSnapshot, mode string) *int {
	t.Helper()
	refreshes := new(int)
	x.options.CurrentTrust = func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		return x.f.a, current, nil
	}
	x.options.RefreshTrust = func(ctx context.Context) error {
		if _, ok := ctx.Deadline(); !ok {
			t.Fatal("node refresh has no deadline")
		}
		*refreshes++
		if mode == "unavailable" {
			return io.ErrUnexpectedEOF
		}
		if mode == "unchanged" {
			return nil
		}
		if mode == "revoked" {
			x.f.snapshot.Payload.Generation++
			x.f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: x.actor.Principal, Mode: "compromise", Reason: "test", FirstGeneration: x.f.snapshot.Payload.Generation}}
			x.f.signSnapshot(t)
		}
		current = x.f.view
		if mode == "concurrent" {
			return fmt.Errorf("concurrent refresh superseded fetch")
		}
		return nil
	}
	if mode == "missing" {
		x.options.RefreshTrust = nil
	}
	var err error
	x.app, err = clusterapp.NewNodeJobs(x.ctx, x.options)
	check(t, err)
	return refreshes
}

func TestNodePrepareSnapshotRollover(t *testing.T) {
	for _, mode := range []string{"refresh", "concurrent", "unchanged", "unavailable", "missing", "signature", "binding", "revoked"} {
		t.Run(mode, func(t *testing.T) {
			x := newNodeJobsFixture(t)
			old := x.f.view
			x.f.snapshot.Payload.Generation++
			x.f.signSnapshot(t)
			refreshes := nodeRolloverView(t, x, old, mode)
			lease, err := x.store.Reserve(x.ctx, x.actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, x.f.now)
			check(t, err)
			signed := signedJobFixture(t, x.f, x.options.Node, lease)
			signed.Manifest.DatasetSelector = "toy"
			request, err := signed.Manifest.SigningRequest()
			check(t, err)
			signed.Proof, err = trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, request, x.f.now)
			check(t, err)
			if mode == "signature" {
				signed.Proof.Signature[0] ^= 1
			}
			if mode == "binding" {
				signed.Proof.Request.Context = "other-job"
			}
			_, err = x.app.Prepare(x.ctx, x.actor, clusterapp.NodePrepareRequest{ExpectedLeaseVersion: lease.Version, Signed: signed})
			success := mode == "refresh" || mode == "concurrent"
			behind := mode == "unchanged" || mode == "unavailable" || mode == "missing"
			if (err == nil) != success || errors.Is(err, trust.ErrTrustBehind) != behind {
				t.Fatalf("unexpected rollover classification: %v", err)
			}
			wantRefresh := 1
			if mode == "signature" || mode == "binding" || mode == "missing" {
				wantRefresh = 0
			}
			if *refreshes != wantRefresh {
				t.Fatalf("refreshes=%d, want %d", *refreshes, wantRefresh)
			}
			if !success {
				active, err := x.store.ActiveExecution(x.ctx)
				check(t, err)
				if active != nil && active.Job != nil {
					t.Fatal("rejected trust mutated prepared job")
				}
			}
		})
	}
}

func TestNodeTransportSnapshotRollover(t *testing.T) {
	for _, evidence := range []string{"proof", "credential", "both"} {
		for _, mode := range []string{"refresh", "unavailable", "signature", "credential_binding", "revoked"} {
			t.Run(evidence+"/"+mode, func(t *testing.T) {
				x := newNodeJobsFixture(t)
				prepared, manifest := x.prepare(t)
				q := x.transportRequest(t, prepared, manifest)
				old := x.f.view
				x.f.snapshot.Payload.Generation++
				x.f.signSnapshot(t)
				if evidence != "credential" {
					request, err := q.Signed.Plan.SigningRequest()
					check(t, err)
					q.Signed.Proof, err = trust.SignPrincipalProof(x.f.a, x.f.client.Chain, x.f.client.Key, x.f.view, request, x.f.now)
					check(t, err)
				}
				if evidence != "proof" {
					q.Credential.Snapshot = x.f.snapshot
				}
				if mode == "signature" {
					q.Signed.Proof.Signature[0] ^= 1
				}
				if mode == "credential_binding" {
					q.Credential.Binding.Attempt = id(t)
				}
				refreshes := nodeRolloverView(t, x, old, mode)
				job, err := x.app.Transport(x.ctx, x.actor, manifest.Job, q)
				if mode == "refresh" {
					check(t, err)
					check(t, x.app.Cleanup(context.Background(), nodeagent.CleanupRequest{Job: job.ID, Attempt: manifest.Attempt, TransportHash: job.TransportHash, Workload: prepared.Request.Request.Scope, WorkloadInitialized: true}))
				} else {
					if err == nil || errors.Is(err, trust.ErrTrustBehind) != (mode == "unavailable") {
						t.Fatalf("unexpected transport classification: %v", err)
					}
					active, err := x.store.ActiveExecution(x.ctx)
					check(t, err)
					if active.Transport != nil || active.Approval != nil {
						t.Fatal("rejected trust activated transport")
					}
				}
				wantRefresh := 1
				if mode == "signature" || mode == "credential_binding" {
					wantRefresh = 0
				}
				if *refreshes != wantRefresh {
					t.Fatalf("refreshes=%d, want %d", *refreshes, wantRefresh)
				}
			})
		}
	}
}
