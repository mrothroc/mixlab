package clusterapp

import (
	"context"
	"fmt"
	"net/url"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/transport/snapshottls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
)

// RefreshRemoteTrust authenticates the authority to the existing pin, verifies
// the public snapshot signature, then installs it monotonically. It cannot
// enroll, renew, revive an expired identity, or replace a root.
func RefreshRemoteTrust(ctx context.Context, p *principal.Store, endpoint string, clock func() time.Time) error {
	if p == nil || clock == nil {
		return fmt.Errorf("principal and clock required")
	}
	s, err := p.View(clock())
	if err != nil {
		return err
	}
	if !slices.Contains(s.Snapshot.Payload.Endpoints.Payload.URLs, endpoint) {
		return fmt.Errorf("authority endpoint absent from pinned signed endpoints")
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
	if err != nil {
		return err
	}
	c, err := snapshottls.HTTPClient(endpoint, func(chain [][]byte, now time.Time) error {
		return trust.VerifyProvisionalSource(a, chain, trust.Authority, now)
	}, clock, 15*time.Second)
	if err != nil {
		return err
	}
	defer c.CloseIdleConnections()
	base, _ := url.Parse(endpoint)
	var next trust.SignedSnapshot
	if err := enrollmentJSON(ctx, c, "GET", base.ResolveReference(&url.URL{Path: "/v1/trust/snapshots/latest"}).String(), nil, &next); err != nil {
		return err
	}
	if _, err := trust.VerifySnapshot(a, next, clock()); err != nil {
		return err
	}
	if !slices.Contains(next.Payload.Endpoints.Payload.URLs, endpoint) || next.Payload.Endpoints.Payload.Audience != s.Snapshot.Payload.Endpoints.Payload.Audience {
		return fmt.Errorf("refreshed authority endpoint/audience mismatch")
	}
	return refreshPrincipal(ctx, p, next, clock())
}

func RenewRemotePrincipal(ctx context.Context, p *principal.Store, endpoint string, clock func() time.Time) error {
	if err := RefreshRemoteTrust(ctx, p, endpoint, clock); err != nil {
		return err
	}
	s, _, err := p.Active(clock())
	if err != nil {
		return err
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
	if err != nil {
		return err
	}
	q, err := p.BeginRenewal(ctx, clock())
	if err != nil {
		return err
	}
	policy, err := managedPrincipalPolicy(p, trust.Authority, clock)
	if err != nil {
		return err
	}
	c, err := policy.HTTPClient(15 * time.Second)
	if err != nil {
		return err
	}
	defer c.CloseIdleConnections()
	base, _ := url.Parse(endpoint)
	var out enrollment.RenewalResult
	if err := enrollmentJSON(ctx, c, "POST", base.ResolveReference(&url.URL{Path: "/v1/trust/principals/renew"}).String(), q, &out); err != nil {
		return err
	}
	if err := enrollment.ValidateRenewalResult(a, s.Chain, q, out, clock()); err != nil {
		return err
	}
	return p.Renew(ctx, out.Chain, out.Snapshot, clock())
}

func RenewLocalPrincipal(ctx context.Context, a *Authority, p *principal.Store, now time.Time) error {
	v, err := a.Current(ctx, now)
	if err != nil {
		return err
	}
	snapshot, err := a.Snapshots.Load(now)
	if err != nil {
		return err
	}
	if err := refreshPrincipal(ctx, p, snapshot, now); err != nil {
		return err
	}
	s, _, err := p.Active(now)
	if err != nil {
		return err
	}
	q, err := p.BeginRenewal(ctx, now)
	if err != nil {
		return err
	}
	out, err := a.Enrollment.Renew(ctx, s.Chain, q, v, now)
	if err != nil {
		return err
	}
	if err := enrollment.ValidateRenewalResult(a.Anchor, s.Chain, q, out, now); err != nil {
		return err
	}
	return p.Renew(ctx, out.Chain, out.Snapshot, now)
}
