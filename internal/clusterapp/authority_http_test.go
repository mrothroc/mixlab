package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/snapshottls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

// Delay the first TLS write on every accepted connection, not just the first
// request. Retrying a too-short timeout cannot bypass this consistently slow
// authority. Close interrupts the delay just as it interrupts a socket write.
type delayedAuthorityListener struct {
	net.Listener
	delay    time.Duration
	accepted atomic.Int32
	started  chan struct{}
	once     sync.Once
}

func (l *delayedAuthorityListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	l.accepted.Add(1)
	return &delayedAuthorityConn{Conn: c, listener: l, closed: make(chan struct{})}, nil
}

type delayedAuthorityConn struct {
	net.Conn
	listener  *delayedAuthorityListener
	writeOnce sync.Once
	closeOnce sync.Once
	closed    chan struct{}
}

func (c *delayedAuthorityConn) Write(b []byte) (int, error) {
	var err error
	c.writeOnce.Do(func() {
		c.listener.once.Do(func() { close(c.listener.started) })
		timer := time.NewTimer(c.listener.delay)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-c.closed:
			err = net.ErrClosed
		}
	})
	if err != nil {
		return 0, err
	}
	return c.Conn.Write(b)
}

func (c *delayedAuthorityConn) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return c.Conn.Close()
}

func TestAuthorityHTTPControllerStartupSlowTLS(t *testing.T) {
	for _, mode := range []string{"startup", "service", "canceled"} {
		cancelDuringTLS := mode == "canceled"
		name := mode + "_signed_snapshot_after_twenty_seconds"
		if cancelDuringTLS {
			name = "cancellation_interrupts_tls"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			l, err := net.Listen("tcp", "127.0.0.1:0")
			check(t, err)
			defer func() { _ = l.Close() }()
			endpoint := "https://" + l.Addr().String()
			p := initializedAt(t, endpoint)
			check(t, InitializeAuthority(ctx, p, testNow))
			a, err := OpenAuthority(ctx, p, testNow)
			check(t, err)
			defer func() { check(t, a.Close()) }()
			now := testNow.Add(24 * time.Hour)
			clock := func() time.Time { return now }
			cp, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "controller")}, statehome.Context{Kind: statehome.Principal})
			check(t, err)
			c, err := principal.Open(cp, now)
			check(t, err)
			defer func() { check(t, c.Close()) }()
			if _, _, err := c.Active(now); err == nil {
				t.Fatal("fixture must start with expired cached trust")
			}
			old, err := c.View(now)
			check(t, err)
			// Real HTTPS intentionally uses wall time: the production HTTP and
			// socket deadlines, signed response, and root pin all remain active.
			slow := &delayedAuthorityListener{Listener: l, delay: 20 * time.Second, started: make(chan struct{})}
			served := make(chan error, 1)
			go func() { served <- ServeAuthority(ctx, slow, a, clock) }()
			defer func() { cancel(); check(t, <-served) }()
			result := make(chan error, 1)
			started := time.Now()
			go func() {
				if mode == "service" {
					result <- refreshPrincipalTrust(ctx, c, clock)
				} else {
					result <- RefreshControllerStartup(ctx, c, endpoint, clock, nil)
				}
			}()
			if cancelDuringTLS {
				select {
				case <-slow.started:
					cancel()
				case <-ctx.Done():
					<-result
					t.Fatal("TLS write never started")
				}
			}
			err = <-result
			if cancelDuringTLS {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancelled startup: %v", err)
				}
				if _, _, err := c.Active(now); err == nil {
					t.Fatal("cancelled startup installed trust")
				}
				return
			}
			check(t, err)
			if time.Since(started) < slow.delay {
				t.Fatal("startup bypassed the delayed TLS write")
			}
			if slow.accepted.Load() != 1 {
				t.Fatalf("slow fetch needed %d connections, want one", slow.accepted.Load())
			}
			current, _, err := c.Active(now)
			check(t, err)
			before, err := json.Marshal(old.Snapshot)
			check(t, err)
			after, err := json.Marshal(current.Snapshot)
			check(t, err)
			if bytes.Equal(before, after) || current.Fingerprint != old.Fingerprint || current.Principal != old.Principal || !bytes.Equal(current.Chain[0], old.Chain[0]) {
				t.Fatal("startup did not refresh signed trust while preserving the pinned identity")
			}
		})
	}
}

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
