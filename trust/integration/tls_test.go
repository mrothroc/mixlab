package integration

import (
	"crypto/ed25519"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type tlsFixture struct {
	mu                     sync.RWMutex
	now                    time.Time
	a                      trust.Anchor
	issuer, snapshotCert   []byte
	issuerKey, snapshotKey ed25519.PrivateKey
	snapshot               trust.SignedSnapshot
	view                   trust.VerifiedSnapshot
	server, client         managedtls.Identity
	serverID, clientID     string
}

func newTLSFixture(t *testing.T) *tlsFixture {
	t.Helper()
	f := &tlsFixture{now: time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)}
	r := key(t)
	root, err := certificates.CreateRoot(id(t), r, f.now)
	check(t, err)
	fp, err := trust.RootFingerprint(r.Public())
	check(t, err)
	f.a, err = trust.PinRoot(root, fp, f.now)
	check(t, err)
	f.issuerKey, f.snapshotKey = key(t), key(t)
	f.issuer, err = certificates.Issue(f.a, root, r, certificates.Issuer, "", "", f.issuerKey.Public(), f.now)
	check(t, err)
	f.snapshotCert, err = certificates.Issue(f.a, root, r, certificates.SnapshotSigner, "", "", f.snapshotKey.Public(), f.now)
	check(t, err)
	f.serverID, f.clientID = id(t), id(t)
	f.server, f.client = f.leaf(t, trust.Authority, f.serverID), f.leaf(t, trust.Controller, f.clientID)
	e, err := trust.SignAuthorityEndpoints(f.a, trust.AuthorityEndpoints{
		Version: trust.EndpointVersion, Cluster: f.a.Cluster(), Audience: "test", URLs: []string{"https://test.invalid"}, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(24 * time.Hour).Unix(),
	}, r, f.now)
	check(t, err)
	f.snapshot = trust.SignedSnapshot{Payload: trust.Snapshot{
		Version: trust.SnapshotVersion, Cluster: f.a.Cluster(), Generation: 1,
		IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(trust.SnapshotLifetime).Unix(),
		Issuers: [][]byte{f.issuer}, EligibleRoles: []trust.Role{trust.Authority, trust.Controller, trust.Coordinator}, Endpoints: e,
	}}
	f.signSnapshot(t)
	return f
}

func (f *tlsFixture) leaf(t *testing.T, role trust.Role, principal string) managedtls.Identity {
	t.Helper()
	k := key(t)
	der, err := certificates.Issue(f.a, f.issuer, f.issuerKey, certificates.Principal, role, principal, k.Public(), f.now)
	check(t, err)
	return managedtls.Identity{Chain: [][]byte{der, f.issuer, f.a.DER()}, Key: k}
}

func (f *tlsFixture) signSnapshot(t *testing.T) {
	t.Helper()
	var err error
	f.snapshot, err = trust.SignSnapshot(f.a, f.snapshot.Payload, f.snapshotCert, f.snapshotKey, f.now)
	check(t, err)
	f.view, err = trust.VerifySnapshot(f.a, f.snapshot, f.now)
	check(t, err)
}
func (f *tlsFixture) clock() time.Time { f.mu.RLock(); defer f.mu.RUnlock(); return f.now }
func (f *tlsFixture) verify(role trust.Role, principal string) managedtls.Verify {
	return func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		p, err := trust.AuthenticatePrincipal(f.a, f.view, chain, now)
		if err != nil {
			return p, err
		}
		if p.Role != role || p.Principal != principal {
			return trust.AuthenticatedPrincipal{}, fmt.Errorf("unexpected peer identity")
		}
		return p, nil
	}
}
func (f *tlsFixture) policies(t *testing.T) (*managedtls.Policy, *managedtls.Policy) {
	t.Helper()
	s, err := managedtls.New(f.server, f.verify(trust.Controller, f.clientID), f.clock)
	check(t, err)
	c, err := managedtls.New(f.client, f.verify(trust.Authority, f.serverID), f.clock)
	check(t, err)
	return s, c
}

func serveManaged(t *testing.T, p *managedtls.Policy, calls *atomic.Int32) *httptest.Server {
	t.Helper()
	h := p.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		identity, ok := managedtls.Principal(r.Context())
		if !ok || identity.Role != trust.Controller {
			t.Error("missing authenticated principal")
			w.WriteHeader(500)
			return
		}
		calls.Add(1)
		_, _ = io.WriteString(w, "authenticated")
	}))
	s := httptest.NewUnstartedServer(h)
	s.TLS = p.ServerConfig()
	s.Config.ErrorLog = log.New(io.Discard, "", 0)
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func clientFor(t *testing.T, c *tls.Config) *http.Client {
	t.Helper()
	tr := &http.Transport{TLSClientConfig: c, TLSHandshakeTimeout: time.Second}
	t.Cleanup(tr.CloseIdleConnections)
	return &http.Client{Transport: tr, Timeout: 2 * time.Second}
}

func getManaged(t *testing.T, c *http.Client, url string) (int, bool) {
	t.Helper()
	reused := false
	r, err := http.NewRequest(http.MethodGet, url, nil)
	check(t, err)
	r = r.WithContext(httptrace.WithClientTrace(r.Context(), &httptrace.ClientTrace{GotConn: func(info httptrace.GotConnInfo) { reused = info.Reused }}))
	response, err := c.Do(r)
	check(t, err)
	_, err = io.Copy(io.Discard, response.Body)
	check(t, err)
	check(t, response.Body.Close())
	return response.StatusCode, reused
}

func TestManagedTLSKeepaliveReauth(t *testing.T) {
	for _, failure := range []string{"revoked", "stale"} {
		t.Run(failure, func(t *testing.T) {
			f := newTLSFixture(t)
			serverPolicy, clientPolicy := f.policies(t)
			var calls atomic.Int32
			s := serveManaged(t, serverPolicy, &calls)
			client := clientFor(t, clientPolicy.ClientConfig())
			status, _ := getManaged(t, client, s.URL)
			if status != 200 {
				t.Fatal(status)
			}
			f.mu.Lock()
			if failure == "stale" {
				f.now = f.now.Add(trust.SnapshotLifetime + time.Second)
			} else {
				f.snapshot.Payload.Generation++
				f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.clientID, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
				f.signSnapshot(t)
			}
			f.mu.Unlock()
			status, reused := getManaged(t, client, s.URL)
			if status != 401 || !reused || calls.Load() != 1 {
				t.Fatalf("status=%d reused=%v calls=%d", status, reused, calls.Load())
			}
		})
	}
}

func TestManagedTLSRejectsInvalidPeers(t *testing.T) {
	for _, problem := range []string{"wrong-root", "wrong-role", "wrong-id", "issuer-key", "missing-cert", "tls12", "missing-alpn", "stale", "expired"} {
		t.Run(problem, func(t *testing.T) {
			f := newTLSFixture(t)
			s, c := f.policies(t)
			var calls atomic.Int32
			server := serveManaged(t, s, &calls)
			cfg := c.ClientConfig()
			switch problem {
			case "wrong-root":
				_, other := newTLSFixture(t).policies(t)
				cfg = other.ClientConfig()
			case "wrong-role", "wrong-id":
				role, principal := trust.Controller, id(t)
				if problem == "wrong-role" {
					role, principal = trust.Coordinator, f.clientID
				}
				p, err := managedtls.New(f.leaf(t, role, principal), f.verify(trust.Authority, f.serverID), f.clock)
				check(t, err)
				cfg = p.ClientConfig()
			case "issuer-key":
				p, err := managedtls.New(managedtls.Identity{Chain: [][]byte{f.issuer, f.issuer, f.a.DER()}, Key: f.issuerKey}, f.verify(trust.Authority, f.serverID), f.clock)
				check(t, err)
				cfg = p.ClientConfig()
			case "missing-cert":
				cfg.Certificates = nil
			case "tls12":
				cfg.MinVersion, cfg.MaxVersion = tls.VersionTLS12, tls.VersionTLS12
			case "missing-alpn":
				cfg.NextProtos = nil
			case "stale", "expired":
				f.mu.Lock()
				if problem == "stale" {
					f.now = f.now.Add(16 * time.Minute)
				} else {
					f.now = f.now.Add(40 * 24 * time.Hour)
				}
				f.mu.Unlock()
			}
			response, err := clientFor(t, cfg).Get(server.URL)
			if response != nil {
				_ = response.Body.Close()
			}
			if err == nil || calls.Load() != 0 {
				t.Fatalf("accepted %s: %v", problem, err)
			}
		})
	}
}

func TestManagedTLSConfigurationAndPlaintext(t *testing.T) {
	f := newTLSFixture(t)
	if _, err := managedtls.New(managedtls.Identity{Chain: f.client.Chain, Key: key(t)}, f.verify(trust.Authority, f.serverID), f.clock); err == nil {
		t.Fatal("mismatched signer")
	}
	if _, err := managedtls.New(f.client, nil, f.clock); err == nil {
		t.Fatal("missing verifier")
	}
	s, c := f.policies(t)
	for _, cfg := range []*tls.Config{s.ServerConfig(), c.ClientConfig()} {
		if !cfg.SessionTicketsDisabled || cfg.ClientSessionCache != nil || cfg.VerifyConnection == nil || cfg.MinVersion != tls.VersionTLS13 || cfg.MaxVersion != tls.VersionTLS13 {
			t.Fatal("weakened TLS config")
		}
	}
	copy := c.ClientConfig()
	copy.Certificates[0].Certificate[0][0] ^= 1
	if c.ClientConfig().Certificates[0].Certificate[0][0] != f.client.Chain[0][0] {
		t.Fatal("configuration aliasing")
	}
	if _, err := c.Authenticate(tls.ConnectionState{}); err == nil {
		t.Fatal("accepted incomplete TLS")
	}
	recorder := httptest.NewRecorder()
	s.AuthenticateHTTP(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { t.Error("plaintext reached handler") })).ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "http://localhost/", nil))
	if recorder.Code != 401 {
		t.Fatal(recorder.Code)
	}
}

func TestManagedTLSClientRevalidatesAndNeverRedirects(t *testing.T) {
	f := newTLSFixture(t)
	s, c := f.policies(t)
	var calls atomic.Int32
	server := serveManaged(t, s, &calls)
	client, err := c.HTTPClient(2 * time.Second)
	check(t, err)
	t.Cleanup(client.CloseIdleConnections)
	if response, err := client.Get("http://127.0.0.1:1/"); err == nil {
		_ = response.Body.Close()
		t.Fatal("plaintext allowed")
	}
	status, _ := getManaged(t, client, server.URL)
	if status != 200 {
		t.Fatal(status)
	}
	_, reused := getManaged(t, client, server.URL)
	if reused {
		t.Fatal("client skipped handshake revalidation")
	}
	f.mu.Lock()
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.serverID, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	f.signSnapshot(t)
	f.mu.Unlock()
	if response, err := client.Get(server.URL); err == nil {
		_ = response.Body.Close()
		t.Fatal("revoked server accepted")
	}
	if calls.Load() != 2 {
		t.Fatal("request sent to revoked server")
	}

	f = newTLSFixture(t)
	s, c = f.policies(t)
	redirects := atomic.Int32{}
	redirect := httptest.NewUnstartedServer(s.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		redirects.Add(1)
		http.Redirect(w, r, "/elsewhere", http.StatusTemporaryRedirect)
	})))
	redirect.TLS = s.ServerConfig()
	redirect.StartTLS()
	t.Cleanup(redirect.Close)
	client, err = c.HTTPClient(2 * time.Second)
	check(t, err)
	t.Cleanup(client.CloseIdleConnections)
	status, _ = getManaged(t, client, redirect.URL)
	if status != http.StatusTemporaryRedirect || redirects.Load() != 1 {
		t.Fatal("followed redirect")
	}
}
