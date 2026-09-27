package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/snapshottls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestAuthorityHTTPRefreshRenewalAndRevocation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	p := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	now := testNow.Add(24 * time.Hour)
	clock := func() time.Time { return now }
	done := make(chan error, 1)
	go func() { done <- ServeAuthority(ctx, l, a, clock) }()
	defer func() { cancel(); check(t, <-done) }()
	cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	c, err := principal.Open(cp, now)
	check(t, err)
	defer func() { _ = c.Close() }()
	if _, _, err := c.Active(now); err == nil {
		t.Fatal("fixture snapshot should be stale")
	}
	check(t, RefreshRemoteTrust(ctx, c, endpoint, clock))
	old, _, err := c.Active(now)
	check(t, err)
	q, err := c.BeginRenewal(ctx, now)
	check(t, err)
	reopened, err := principal.Open(cp, now)
	check(t, err)
	q2, err := reopened.BeginRenewal(ctx, now)
	check(t, err)
	check(t, reopened.Close())
	b1, _ := json.Marshal(q)
	b2, _ := json.Marshal(q2)
	if !bytes.Equal(b1, b2) {
		t.Fatal("lost-response renewal retry changed nonce")
	}
	check(t, RenewRemotePrincipal(ctx, c, endpoint, clock))
	current, _, err := c.Active(now)
	check(t, err)
	if bytes.Equal(current.Chain[0], old.Chain[0]) || current.Principal != old.Principal || current.Key.ID != old.Key.ID {
		t.Fatal("renewal failed to preserve identity/change certificate")
	}
	// Public snapshot traffic cannot reach renewal, even if a caller tries to
	// repurpose the route-confined client. Anonymous raw HTTP is rejected too.
	public, err := snapshottls.HTTPClient(endpoint, func(chain [][]byte, now time.Time) error {
		return trust.VerifyProvisionalSource(a.Anchor, chain, trust.Authority, now)
	}, clock, time.Second)
	check(t, err)
	defer public.CloseIdleConnections()
	r, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/trust/principals/renew", bytes.NewReader(b1))
	check(t, err)
	if _, err := public.Do(r); err == nil {
		t.Fatal("snapshot adapter reached application route")
	}
	rawTransport := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS13, NextProtos: []string{"http/1.1"}, Time: clock}, DisableKeepAlives: true}
	defer rawTransport.CloseIdleConnections()
	res, err := (&http.Client{Transport: rawTransport, Timeout: time.Second}).Do(r)
	check(t, err)
	_ = res.Body.Close()
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("anonymous renewal status %d", res.StatusCode)
	}
	_, err = a.Snapshots.Revoke(ctx, "principal", current.Principal, "prospective", "test", now)
	check(t, err)
	check(t, RefreshRemoteTrust(ctx, c, endpoint, clock))
	if _, _, err := c.Active(now); err == nil {
		t.Fatal("public refresh reinstated revoked principal")
	}
	if err := RenewRemotePrincipal(ctx, c, endpoint, clock); err == nil {
		t.Fatal("revoked principal renewed")
	}
}

func TestLocalAuthorityRenewalUsesSameDurableContract(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	c, err := principal.Open(cp, testNow)
	check(t, err)
	defer func() { _ = c.Close() }()
	old, err := c.View(testNow)
	check(t, err)
	now := testNow.Add(time.Hour)
	check(t, RenewLocalPrincipal(ctx, a, c, now))
	r, _, err := c.Active(now)
	check(t, err)
	if r.Principal != old.Principal || r.Key.ID != old.Key.ID || bytes.Equal(r.Chain[0], old.Chain[0]) {
		t.Fatal("local renewal changed identity or did not renew")
	}
}
