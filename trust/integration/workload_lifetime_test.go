package integration

import (
	"testing"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestMultiDayWorkloadTransportExpiryAndRevocation(t *testing.T) {
	f := newTLSFixture(t)
	m := signedJobFixture(t, f, id(t), nodeagent.Lease{ID: id(t), Run: id(t)}).Manifest
	m.Limits.RuntimeSeconds = 3 * 24 * 3600
	deadline, err := m.WorkloadDeadline()
	check(t, err)
	signed, identities := transportFixtureUntil(t, f, m, deadline)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	_, err = grouptransport.Accept(f.a, f.view, actor, m, identities[0].Chain, signed, f.now)
	check(t, err)
	peer := signed.Plan.Members[0]
	_, _, err = certificates.Verify(f.a, peer.Chain[0], peer.Chain[1], certificates.Principal, f.now.Add(48*time.Hour))
	check(t, err)
	if _, _, err := certificates.Verify(f.a, peer.Chain[0], peer.Chain[1], certificates.Principal, deadline); err == nil {
		t.Fatal("accepted expired multi-day certificate")
	}
	// An authentic, correctly signed plan may still exceed the admitted job.
	tooLong, keys := transportFixtureUntil(t, f, m, deadline.Add(time.Second))
	if _, err := grouptransport.Accept(f.a, f.view, actor, m, keys[0].Chain, tooLong, f.now); err == nil {
		t.Fatal("accepted lifetime beyond declared runtime and cleanup")
	}
	// Long-lived certificates still require fresh snapshots and obey revocation.
	f.now = f.now.Add(2 * time.Hour)
	if err := trust.VerifyWorkload(f.a, f.view, peer.Chain, peer.Binding, f.now); err == nil {
		t.Fatal("accepted stale snapshot for long-lived workload")
	}
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.IssuedAt = f.now.Unix()
	f.snapshot.Payload.ExpiresAt = f.now.Add(trust.SnapshotLifetime).Unix()
	f.signSnapshot(t)
	check(t, trust.VerifyWorkload(f.a, f.view, peer.Chain, peer.Binding, f.now))
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: peer.Binding.Principal, Mode: "compromise", FirstGeneration: f.snapshot.Payload.Generation, Reason: "test"}}
	f.signSnapshot(t)
	if err := trust.VerifyWorkload(f.a, f.view, peer.Chain, peer.Binding, f.now); err == nil {
		t.Fatal("long-lived workload bypassed revocation")
	}
}
