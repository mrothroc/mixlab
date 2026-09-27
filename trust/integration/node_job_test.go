package integration

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerhost"
)

func signedJobFixture(t *testing.T, f *tlsFixture, node string, lease nodeagent.Lease) nodejob.Signed {
	t.Helper()
	members := []distributed.DDPGroupMember{{MemberID: id(t), Rank: 0}, {MemberID: id(t), Rank: 1}}
	membership, err := distributed.NewDDPGroupMembership(lease.Run, id(t), 1, "ring", members)
	check(t, err)
	hash := strings.Repeat("a", 64)
	config := json.RawMessage(`{"model_dim":16}`)
	m := nodejob.Manifest{Version: nodejob.Version, Job: id(t), Attempt: id(t), Lease: lease.ID, Node: node, Controller: f.clientID, Nonce: id(t), Membership: membership, Members: []nodejob.Member{{Node: node, MemberID: members[0].MemberID, Rank: 0}, {Node: id(t), MemberID: members[1].MemberID, Rank: 1}}, Rank: 0, BuildID: hash, Config: config, ConfigHash: nodejob.Hash(config), ProgramHash: hash, WeightLayoutHash: hash, OptimizerHash: hash, DatasetSelector: "toy.train", DatasetID: hash, Artifacts: []nodejob.ArtifactRef{}, Mode: "arch", Transport: "tls13-ring", Limits: nodejob.Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Created: f.now.Unix(), Expires: f.now.Unix() + 60}
	q, err := m.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	return nodejob.Signed{Manifest: m, Proof: proof}
}

func TestSignedNodeJobDurablePrepareAndCancellation(t *testing.T) {
	ctx := context.Background()
	f := newTLSFixture(t)
	node := id(t)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	check(t, err)
	s, err := nodeagent.Initialize(ctx, p, f.a.Cluster(), node, 1)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	lease, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	before, err := p.ReadFileLimit("node-leases.json", 1<<20)
	check(t, err)
	signed := signedJobFixture(t, f, node, lease)
	accepted, err := nodejob.Accept(f.a, f.view, actor, node, signed, f.now)
	check(t, err)
	j, err := s.PrepareJob(ctx, actor, accepted, lease.Version, f.now)
	check(t, err)
	if j.State != nodeagent.JobPrepared {
		t.Fatal(j)
	}
	// Fault injection: restore the lease bytes from immediately before its
	// publication, leaving the already-durable job acceptance in place.
	check(t, p.WriteFile("node-leases.json", before))
	s, err = nodeagent.Open(p, f.a.Cluster(), node)
	check(t, err)
	j2, err := s.PrepareJob(ctx, actor, accepted, lease.Version, f.now)
	check(t, err)
	if j2 != j {
		t.Fatal("prepare retry changed job")
	}
	availability, err := s.Availability(f.now)
	check(t, err)
	if availability.Available || availability.Lease.Job != j.ID || availability.Lease.State != nodeagent.Prepared {
		t.Fatal("job/lease binding not recovered", availability)
	}
	// A newer trust generation must not rewrite the original commit evidence
	// or make an otherwise exact prepare retry fail.
	f.snapshot.Payload.Generation++
	f.signSnapshot(t)
	accepted, err = nodejob.Accept(f.a, f.view, actor, node, signed, f.now)
	check(t, err)
	j2, err = s.PrepareJob(ctx, actor, accepted, lease.Version, f.now)
	check(t, err)
	if j2 != j {
		t.Fatal("snapshot refresh changed idempotent outcome")
	}
	other := actor
	other.Principal = id(t)
	if _, err := s.JobStatus(other, j.ID, f.now); err == nil {
		t.Fatal("other controller inspected job")
	}
	if _, err := s.CancelJob(ctx, other, j.ID, j.Version, f.now); err == nil {
		t.Fatal("other controller canceled job")
	}
	j, err = s.CancelJob(ctx, actor, j.ID, j.Version, f.now)
	check(t, err)
	if !j.CancelRequested || j.State != nodeagent.JobPrepared {
		t.Fatal("cancel claimed terminal outcome", j)
	}
	if err := s.ReleaseUnprepared(ctx, lease.ID); err == nil {
		t.Fatal("released bound job without host confirmation")
	}
	availability, err = s.Availability(f.now)
	check(t, err)
	if availability.Available {
		t.Fatal("cancel intent freed accelerator")
	}
	workerDir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(workerDir, 0700))
	wp, err := statehome.Resolve(statehome.Options{ExactDir: workerDir}, statehome.Context{Kind: statehome.Worker})
	check(t, err)
	host, err := workerhost.NewAttemptStore(wp)
	check(t, err)
	fence, err := s.PreparationFence(ctx, j.ID)
	check(t, err)
	outcome, err := host.FencePreparation(ctx, fence)
	check(t, err)
	j, err = s.ApplyOutcome(ctx, outcome)
	check(t, err)
	if j.State != nodeagent.JobCanceled {
		t.Fatal("prepared cancellation not terminal", j)
	}
	check(t, s.CompleteCleanup(ctx, j.ID, func(context.Context, nodeagent.CleanupRequest) error { return nil }))
	availability, err = s.Availability(f.now)
	check(t, err)
	if !availability.Available {
		t.Fatal("fenced cancellation did not release node")
	}
}

func TestSignedNodeJobRejectsTamperingAndWrongActor(t *testing.T) {
	f := newTLSFixture(t)
	node := id(t)
	lease := nodeagent.Lease{ID: id(t), Run: id(t)}
	signed := signedJobFixture(t, f, node, lease)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	for _, test := range []string{"config", "controller", "target", "expired", "signature"} {
		t.Run(test, func(t *testing.T) {
			b, err := json.Marshal(signed)
			check(t, err)
			var copy nodejob.Signed
			check(t, json.Unmarshal(b, &copy))
			who := actor
			target := node
			now := f.now
			switch test {
			case "config":
				copy.Manifest.Config = json.RawMessage(`{"model_dim":32}`)
				copy.Manifest.ConfigHash = nodejob.Hash(copy.Manifest.Config)
			case "controller":
				who.Principal = id(t)
			case "target":
				target = id(t)
			case "expired":
				now = now.Add(61e9)
			case "signature":
				copy.Proof.Signature[0] ^= 1
			}
			if _, err := nodejob.Accept(f.a, f.view, who, target, copy, now); err == nil {
				t.Fatal("accepted invalid job")
			}
		})
	}
}
