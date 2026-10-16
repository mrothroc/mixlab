//go:build darwin || linux

package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

// This test-owned proxy uses the enrolled node's real identity on the inbound
// connection and the exact authenticated controller/coordinator on the outbound
// connection. Only retry-authorize responses are modified, never recovery
// evidence. Neither production TLS verification nor application auth is relaxed.
type compositionDeadlineProxy struct {
	endpoint string
	mu       sync.Mutex
	prepared *clusterapp.NodeLearnerRetryPrepared
	starts   int
	err      error
}

func compositionNodeHTTPClient(t *testing.T, p *principal.Store, node string) *http.Client {
	t.Helper()
	s, key, err := p.Active(time.Now())
	compositionCheck(t, err)
	a, err := trust.PinRoot(s.Root, s.Fingerprint, time.Now())
	compositionCheck(t, err)
	policy, err := managedtls.New(managedtls.Identity{Chain: s.Chain, Key: key}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		current, err := p.View(now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		view, err := trust.VerifySnapshot(a, current.Snapshot, now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		peer, err := trust.AuthenticatePrincipal(a, view, chain, now)
		if err == nil && (peer.Role != trust.Node || peer.Principal != node) {
			err = fmt.Errorf("proxy upstream is not the enrolled node")
		}
		return peer, err
	}, time.Now)
	compositionCheck(t, err)
	client, err := policy.HTTPClient(time.Minute)
	compositionCheck(t, err)
	t.Cleanup(client.CloseIdleConnections)
	return client
}

func newCompositionDeadlineProxy(t *testing.T, node, controller, coordinatorPath statehome.Path, upstream string) *compositionDeadlineProxy {
	t.Helper()
	open := func(path statehome.Path) *principal.Store {
		p, err := principal.Open(path, time.Now())
		compositionCheck(t, err)
		t.Cleanup(func() { compositionCheck(t, p.Close()) })
		return p
	}
	np := open(node)
	identity, _, err := np.Active(time.Now())
	compositionCheck(t, err)
	clients := map[string]*http.Client{}
	for _, path := range []statehome.Path{controller, coordinatorPath} {
		p := open(path)
		s, _, err := p.Active(time.Now())
		compositionCheck(t, err)
		clients[s.Principal] = compositionNodeHTTPClient(t, p, identity.Principal)
	}
	policy, err := clusterapp.NodePrincipalPolicy(np, time.Now)
	compositionCheck(t, err)
	x := &compositionDeadlineProxy{}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, status, err := x.forward(r, clients, upstream, np)
		if err != nil {
			if r.Context().Err() != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) {
				return
			}
			x.mu.Lock()
			x.err = err
			x.mu.Unlock()
			http.Error(w, "test proxy failed", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write(body)
	})
	server := httptest.NewUnstartedServer(policy.AuthenticateHTTP(handler))
	server.TLS = policy.ServerConfig()
	server.StartTLS()
	t.Cleanup(server.Close)
	x.endpoint = server.Listener.Addr().String()
	return x
}

func (x *compositionDeadlineProxy) forward(r *http.Request, clients map[string]*http.Client, upstream string, node *principal.Store) ([]byte, int, error) {
	actor, ok := managedtls.Principal(r.Context())
	if !ok || clients[actor.Principal] == nil {
		return nil, 0, fmt.Errorf("unexpected proxy caller")
	}
	if strings.HasSuffix(r.URL.Path, "/retry-start") {
		x.mu.Lock()
		x.starts++
		x.mu.Unlock()
	}
	q, err := http.NewRequestWithContext(r.Context(), r.Method, "https://"+upstream+r.URL.RequestURI(), r.Body)
	if err != nil {
		return nil, 0, err
	}
	q.Header = r.Header.Clone()
	q.ContentLength = r.ContentLength
	response, err := clients[actor.Principal].Do(q)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, 2<<20))
	if err != nil || response.StatusCode != http.StatusOK || !strings.HasSuffix(r.URL.Path, "/retry-authorize") {
		return body, response.StatusCode, err
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.prepared == nil {
		var out clusterapp.NodeLearnerRetryPrepared
		if err := json.Unmarshal(body, &out); err != nil {
			return nil, 0, err
		}
		if err := compositionExtendDeadline(node, &out.Request); err != nil {
			return nil, 0, err
		}
		x.prepared = &out
	}
	// Preserve exact adversarial bytes across any response replay too.
	body, err = json.Marshal(x.prepared)
	return body, response.StatusCode, err
}

func compositionExtendDeadline(node *principal.Store, request *workload.SignedRequest) error {
	scope := request.Request.Scope
	scope.Deadline++
	// The CSR binds the scope hash, so a valid adversarial request must regenerate
	// it as well as the node proof. No other scope field or retry evidence changes.
	_, transport, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return err
	}
	changed, err := workload.NewRequest(scope, transport)
	if err != nil {
		return err
	}
	q, err := changed.SigningRequest()
	if err != nil {
		return err
	}
	now := time.Now()
	s, key, err := node.Active(now)
	if err != nil {
		return err
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, now)
	if err != nil {
		return err
	}
	v, err := trust.VerifySnapshot(a, s.Snapshot, now)
	if err != nil {
		return err
	}
	proof, err := trust.SignPrincipalProof(a, s.Chain, key, v, q, now)
	if err != nil {
		return err
	}
	if _, err := trust.VerifyProof(a, v, proof, q, now); err != nil {
		return err
	}
	*request = workload.SignedRequest{Request: changed, Proof: proof}
	return nil
}

func TestInstalledLearnerRecoveryRejectsExtendedDeadline(t *testing.T) {
	x := newInstalledRecoveryFixture(t, true)
	budget := newCompositionBudget(t)
	var diagnosticDeadline time.Time
	for {
		s, err := x.repo.Load(x.ctx, x.submission.Launch.Run)
		if !errors.Is(err, coordinator.ErrNotFound) {
			compositionCheck(t, err)
		}
		budget.observe(t, x, s.Phase)
		awaitingDiagnostic := false
		for _, proxy := range x.proxies {
			proxy.mu.Lock()
			prepared, starts, proxyErr := proxy.prepared, proxy.starts, proxy.err
			proxy.mu.Unlock()
			compositionCheck(t, proxyErr)
			if prepared == nil {
				continue
			}
			scope := prepared.Request.Request.Scope
			if _, err := os.Stat(x.grantPath(scope)); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("extended-deadline recovery grant persisted: stat=%v", err)
			}
			if starts != 0 {
				t.Fatal("extended-deadline retry reached node Start")
			}
			if s.Phase != coordinator.Failed {
				continue
			}
			if len(s.Attempts) != 1 || s.Attempts[0].AttemptID != scope.Attempt {
				t.Fatalf("did not reach installed deadline guard after binding: %+v; %s", s, x.diagnostics())
			}
			// Failure persistence precedes the service event (with an inbox write
			// between them). Observe both without racing that publication order.
			if !strings.Contains(x.diagnostics(), "recovery cannot extend original workload deadline") {
				if diagnosticDeadline.IsZero() {
					diagnosticDeadline = time.Now().Add(10 * time.Second)
				}
				if time.Now().After(diagnosticDeadline) {
					t.Fatalf("failed without installed deadline diagnostic: %s", x.diagnostics())
				}
				awaitingDiagnostic = true
				continue
			}
			b, err := x.authority.ReadFileLimit("workload-issuance.json", 32<<20)
			compositionCheck(t, err)
			var journal struct {
				Entries []struct {
					Grant workload.Grant `json:"grant"`
				} `json:"entries"`
			}
			compositionCheck(t, json.Unmarshal(b, &journal))
			for _, entry := range journal.Entries {
				if entry.Grant.Request.Request.Scope.Attempt == scope.Attempt {
					t.Fatal("authority issued extended-deadline retry credential")
				}
			}
			t.Log("authenticated node request rejected by installed deadline guard before grant, issue or retry start")
			return
		}
		if s.Phase == coordinator.Failed && !awaitingDiagnostic {
			t.Fatalf("failed before adversarial request: %s; %s", s.Failure, x.diagnostics())
		}
		budget.wait(t, x, s.Phase)
	}
}
