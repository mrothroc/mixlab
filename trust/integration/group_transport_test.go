package integration

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func signedTransportFixture(t *testing.T, f *tlsFixture, m nodejob.Manifest) grouptransport.Signed {
	s, _ := transportFixture(t, f, m)
	return s
}
func transportFixture(t *testing.T, f *tlsFixture, m nodejob.Manifest) (grouptransport.Signed, []ringtls.Identity) {
	return transportFixtureUntil(t, f, m, f.now.Add(time.Minute))
}
func transportFixtureUntil(t *testing.T, f *tlsFixture, m nodejob.Manifest, deadline time.Time) (grouptransport.Signed, []ringtls.Identity) {
	t.Helper()
	if !slices.Contains(f.snapshot.Payload.EligibleRoles, trust.Worker) {
		f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Worker)
	}
	f.signSnapshot(t)
	p := grouptransport.Plan{Version: grouptransport.Version, TLSPolicy: grouptransport.TLSPolicy, Cluster: f.a.Cluster(), Controller: f.clientID, Attempt: m.Attempt, Nonce: id(t), Membership: m.Membership, Created: f.now.Unix(), Expires: deadline.Unix()}
	identities := []ringtls.Identity{}
	for i, member := range m.Members {
		job := id(t)
		hash := m.ConfigHash
		if i == m.Rank {
			job = m.Job
			q, e := m.SigningRequest()
			check(t, e)
			hash = q.Digest
		}
		b := trust.WorkloadBinding{Version: trust.WorkloadBindingVersion, Cluster: f.a.Cluster(), Role: trust.Worker, Principal: id(t), Participant: member.Node, Run: m.Membership.RunID, Job: job, Attempt: m.Attempt, Audience: grouptransport.Audience, IssuedAt: f.now.Unix(), ExpiresAt: deadline.Unix(), Group: m.Membership.GroupID, Generation: m.Membership.Generation, MembershipHash: m.Membership.MembersHash, Member: member.MemberID, Rank: i}
		b.Lease, b.ManifestHash = id(t), hash
		if i == m.Rank {
			b.Lease = m.Lease
		}
		private := key(t)
		der, err := certificates.IssueWorkload(f.a, f.issuer, f.issuerKey, private.Public(), b, deadline, f.now)
		check(t, err)
		p.Members = append(p.Members, grouptransport.Member{Node: member.Node, Job: job, MemberID: member.MemberID, Rank: i, ManifestHash: hash, Endpoint: fmt.Sprintf("127.0.0.1:%d", 32000+i), CertificateHash: nodejob.Hash(der), Chain: [][]byte{der, f.issuer, f.a.DER()}, Binding: b})
		p.LoopbackPorts = append(p.LoopbackPorts, 33000+i)
		identities = append(identities, ringtls.Identity{Chain: [][]byte{der, f.issuer, f.a.DER()}, Key: private})
	}
	q, err := p.SigningRequest()
	check(t, err)
	proof, err := trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	return grouptransport.Signed{Plan: p, Proof: proof}, identities
}

func TestSignedTransportExactAdmission(t *testing.T) {
	f := newTLSFixture(t)
	node := id(t)
	job := signedJobFixture(t, f, node, nodeagent.Lease{ID: id(t), Run: id(t)}).Manifest
	signed := signedTransportFixture(t, f, job)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	local := signed.Plan.Members[job.Rank].Chain
	accepted, err := grouptransport.Accept(f.a, f.view, actor, job, local, signed, f.now)
	check(t, err)
	_, _, got, err := accepted.Value()
	check(t, err)
	if got != job.Job {
		t.Fatal("lost job binding")
	}
	for _, name := range []string{"endpoint", "rank", "member", "node", "job", "manifest", "fingerprint", "attempt", "local-key", "controller", "expiry", "signature", "revocation"} {
		t.Run(name, func(t *testing.T) {
			b, err := json.Marshal(signed)
			check(t, err)
			var s grouptransport.Signed
			check(t, json.Unmarshal(b, &s))
			chain := local
			who := actor
			now := f.now
			view := f.view
			switch name {
			case "endpoint":
				s.Plan.Members[0].Endpoint = "0.0.0.0:33000"
			case "rank":
				s.Plan.Members[0].Rank = 1
			case "member":
				s.Plan.Members[0].MemberID = id(t)
			case "node":
				s.Plan.Members[0].Node = id(t)
			case "job":
				s.Plan.Members[0].Job = id(t)
			case "manifest":
				s.Plan.Members[0].ManifestHash = job.ConfigHash
			case "fingerprint":
				s.Plan.Members[0].CertificateHash = job.ConfigHash
			case "attempt":
				s.Plan.Attempt = id(t)
			case "local-key":
				chain = [][]byte{f.client.Chain[0], f.issuer, f.a.DER()}
			case "controller":
				who.Principal = id(t)
			case "expiry":
				now = now.Add(time.Minute)
			case "signature":
				s.Proof.Signature[0] ^= 1
			case "revocation":
				snapshot := f.snapshot.Payload
				snapshot.Generation++
				snapshot.Revocations = []trust.Revocation{{Kind: "principal", ID: s.Plan.Members[0].Binding.Principal, Mode: "compromise", FirstGeneration: snapshot.Generation, Reason: "test"}}
				ss, e := trust.SignSnapshot(f.a, snapshot, f.snapshotCert, f.snapshotKey, f.now)
				check(t, e)
				view, e = trust.VerifySnapshot(f.a, ss, f.now)
				check(t, e)
			}
			if _, err := grouptransport.Accept(f.a, view, who, job, chain, s, now); err == nil {
				t.Fatal("invalid plan admitted")
			}
		})
	}
}

func TestNodeExecutionIntentOutcomeAndCleanup(t *testing.T) {
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
	l, err := s.Reserve(ctx, actor, nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}, f.now)
	check(t, err)
	signed := signedJobFixture(t, f, node, l)
	accepted, err := nodejob.Accept(f.a, f.view, actor, node, signed, f.now)
	check(t, err)
	j, err := s.PrepareJob(ctx, actor, accepted, l.Version, f.now)
	check(t, err)
	m := signed.Manifest
	v, err := distributed.NewLocalGroupView(m.Membership, m.Members[m.Rank].MemberID, m.Rank, m.Attempt)
	check(t, err)
	a := workerjob.Assignment{Version: workerjob.Version, JobID: m.Job, AttemptID: m.Attempt, BuildID: m.BuildID, View: v, Config: m.Config, DatasetSelector: m.DatasetSelector, TrainPattern: "/unused/train.bin", DatasetSHA256: m.DatasetID, ProgramSHA256: m.ProgramHash, RuntimeSeconds: m.Limits.RuntimeSeconds, RingAddresses: [][]string{{"127.0.0.1:33000"}, {"127.0.0.1:33001"}}}
	a.WeightLayoutSHA256, a.OptimizerSHA256 = m.WeightLayoutHash, m.OptimizerHash
	q := nodeagent.StartCommand{IdempotencyKey: id(t), Job: j.ID, ExpectedVersion: j.Version}
	if _, err := s.StartJob(ctx, actor, q, a, f.now); err == nil {
		t.Fatal("started before transport activation")
	}
	sp := signedTransportFixture(t, f, m)
	tp, err := grouptransport.Accept(f.a, f.view, actor, m, sp.Plan.Members[m.Rank].Chain, sp, f.now)
	check(t, err)
	j, err = s.ActivateTransport(ctx, actor, tp, j.Version, f.now)
	check(t, err)
	q.ExpectedVersion = j.Version
	approved, err := s.StartJob(ctx, actor, q, a, f.now)
	check(t, err)
	check(t, approved.Validate())
	av, err := s.Availability(f.now)
	check(t, err)
	if av.Lease.State != nodeagent.Prepared {
		t.Fatal("start intent falsely claims running")
	}
	j, err = s.JobStatus(actor, j.ID, f.now)
	check(t, err)
	if j.State != nodeagent.JobStarting {
		t.Fatal(j)
	}
	_, err = s.StartJob(ctx, actor, q, a, f.now)
	check(t, err)
	a.ProgramSHA256 = m.ConfigHash
	if _, err := s.StartJob(ctx, actor, q, a, f.now); err == nil {
		t.Fatal("changed start accepted")
	}
	oldLease, err := p.ReadFileLimit("node-leases.json", 1<<20)
	check(t, err)
	o := contract.Outcome{JobID: j.ID, AttemptID: j.Attempt, ManifestHash: j.ManifestHash, ApprovalHash: j.ApprovalHash, Version: 1, Kind: contract.Started, PID: 123}
	j, err = s.ApplyOutcome(ctx, o)
	check(t, err)
	check(t, p.WriteFile("node-leases.json", oldLease))
	j, err = s.ApplyOutcome(ctx, o)
	check(t, err)
	av, err = s.Availability(f.now)
	check(t, err)
	if j.State != nodeagent.JobRunning || av.Lease.State != nodeagent.Running {
		t.Fatal("outcome retry failed to repair lease")
	}
	if err := s.CompleteCleanup(ctx, j.ID, func(context.Context, nodeagent.CleanupRequest) error { return nil }); err == nil {
		t.Fatal("released live child")
	}
	o.Version = 2
	o.Kind = contract.Exited
	o.NoChild = true
	j, err = s.ApplyOutcome(ctx, o)
	check(t, err)
	if err := s.CompleteCleanup(ctx, j.ID, func(context.Context, nodeagent.CleanupRequest) error { return errors.New("transport still open") }); err == nil {
		t.Fatal("ignored cleanup failure")
	}
	av, err = s.Availability(f.now)
	check(t, err)
	if av.Available || av.Lease.State != nodeagent.Releasing {
		t.Fatal("premature availability")
	}
	check(t, s.CompleteCleanup(ctx, j.ID, func(_ context.Context, q nodeagent.CleanupRequest) error {
		if q.Job != j.ID || q.Attempt != j.Attempt || q.TransportHash != j.TransportHash {
			t.Error("wrong cleanup binding")
		}
		return nil
	}))
	av, err = s.Availability(f.now)
	check(t, err)
	if !av.Available {
		t.Fatal("terminal cleanup did not release")
	}
	_, err = s.ApplyOutcome(ctx, o)
	check(t, err)
	o.Kind = contract.Failed
	if _, err := s.ApplyOutcome(ctx, o); err == nil {
		t.Fatal("changed terminal outcome")
	}
}
