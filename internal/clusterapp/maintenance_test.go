package clusterapp

import (
	"context"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust/enrollment"
)

func TestAuthorityRenewalSurvivesOriginalLeafExpiry(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	scheduler, err := PrincipalScheduler(a.principalPath, a.Principal, func(ctx context.Context, now time.Time) error { return RenewLocalPrincipal(ctx, a, a.Principal, now) })
	check(t, err)
	at := testNow.Add(20 * 24 * time.Hour)
	_, err = a.Current(ctx, at)
	check(t, err)
	state, err := scheduler.Tick(ctx, at)
	check(t, err)
	if state.Outcome != "renewed" {
		t.Fatal(state)
	}
	// The service was constructed with the old chain. New approvals must use
	// the renewed chain, not that expired captured credential.
	at = testNow.Add(31 * 24 * time.Hour)
	v, err := a.Current(ctx, at)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, v, at)
	check(t, err)
	defer i.Clear()
	_, k, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	x, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	q, err := enrollment.NewRequest(i, k, x.PublicKey().Bytes(), at)
	check(t, err)
	r, err := a.Enrollment.Consume(ctx, q, i.Secret, v, at)
	check(t, err)
	_, err = enrollment.ValidateResult(i, q, r, at)
	check(t, err)
}
