package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

// Keep the relay/gateway acceptance view independent of the signing and TLS
// fixtures so a shared fixture view cannot silently hide the rollover race.
type rolloverTrust struct {
	mu   sync.Mutex
	view trust.VerifiedSnapshot
}

func (s *rolloverTrust) Current(context.Context, time.Time) (trust.VerifiedSnapshot, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.view, nil
}

func (s *rolloverTrust) install(v trust.VerifiedSnapshot) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.view = v
}

func rolloverSnapshot(t *testing.T, f *tlsFixture, generation uint64, revocations []trust.Revocation) trust.VerifiedSnapshot {
	t.Helper()
	f.mu.RLock()
	defer f.mu.RUnlock()
	payload := f.snapshot.Payload
	payload.Generation = generation
	payload.Revocations = revocations
	signed, err := trust.SignSnapshot(f.a, payload, f.snapshotCert, f.snapshotKey, f.now)
	check(t, err)
	v, err := trust.VerifySnapshot(f.a, signed, f.now)
	check(t, err)
	return v
}

func advanceRolloverSigner(t *testing.T, f *tlsFixture) trust.VerifiedSnapshot {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	f.snapshot.Payload.Generation++
	f.signSnapshot(t)
	return f.view
}

type rolloverRelayWire struct {
	mu        sync.Mutex
	requests  [][]byte
	replies   [][]byte
	transfers int
}

func observeRolloverRelay(t *testing.T, x *relayFixture, grantSigner ...managedtls.Identity) (string, *rolloverRelayWire) {
	t.Helper()
	if len(grantSigner) > 1 {
		t.Fatal("at most one alternate grant signer is supported")
	}
	h := retryRelayHandler(t, x)
	artifacts, err := x.g.Handler()
	check(t, err)
	wire := &rolloverRelayWire{}
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != learnerapi.Route {
			wire.mu.Lock()
			wire.transfers++
			wire.mu.Unlock()
			artifacts.ServeHTTP(w, r)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			if r.Context().Err() == nil {
				t.Error(err)
			}
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		recorder := httptest.NewRecorder()
		h.ServeHTTP(capturedLearnerResponse{recorder, w}, r)
		if recorder.Code == http.StatusOK && len(grantSigner) == 1 {
			var response learnerapi.Response
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil || response.Grant == nil {
				t.Errorf("cannot re-sign grant response: grant=%v err=%v", response.Grant, err)
				http.Error(w, "invalid fixture grant", http.StatusInternalServerError)
				return
			}
			// Change only the proof signer, not the actual TLS server certificate
			// or the semantic grant bound to the assignment.
			x.f.mu.RLock()
			proof, err := trust.SignPrincipalProof(x.f.a, grantSigner[0].Chain, grantSigner[0].Key, x.f.view, response.Grant.Proof.Request, x.f.now)
			x.f.mu.RUnlock()
			if err != nil {
				t.Error(err)
				http.Error(w, "fixture grant signing failed", http.StatusInternalServerError)
				return
			}
			response.Grant.Proof = proof
			encoded, err := json.Marshal(response)
			if err != nil {
				t.Error(err)
				http.Error(w, "fixture grant encoding failed", http.StatusInternalServerError)
				return
			}
			recorder.Body.Reset()
			_, _ = recorder.Body.Write(encoded)
		}
		if recorder.Code != http.StatusOK {
			// The final retry may exhaust the caller budget while the gateway
			// is working. Only successful grants are evidence of local catch-up.
			if r.Context().Err() == nil {
				t.Errorf("grant request failed before relay verification: HTTP %d: %s", recorder.Code, recorder.Body)
			}
			w.WriteHeader(recorder.Code)
			_, _ = w.Write(recorder.Body.Bytes())
			return
		}
		wire.mu.Lock()
		wire.requests = append(wire.requests, bytes.Clone(body))
		wire.replies = append(wire.replies, bytes.Clone(recorder.Body.Bytes()))
		wire.mu.Unlock()
		w.WriteHeader(recorder.Code)
		_, _ = w.Write(recorder.Body.Bytes())
	}))
	return server.URL, wire
}

func (w *rolloverRelayWire) assert(t *testing.T, attempts, transfers int, generation uint64) {
	t.Helper()
	w.mu.Lock()
	defer w.mu.Unlock()
	w.assertLocked(t, attempts, transfers, generation)
}

func (w *rolloverRelayWire) assertLocked(t *testing.T, attempts, transfers int, generation uint64) {
	t.Helper()
	if len(w.requests) != attempts || w.transfers != transfers {
		t.Fatalf("unexpected grant retries/transfers: requests=%d transfers=%d want=%d/%d", len(w.requests), w.transfers, attempts, transfers)
	}
	for i := range w.requests {
		if !bytes.Equal(w.requests[0], w.requests[i]) || !bytes.Equal(w.replies[0], w.replies[i]) {
			t.Fatalf("retry %d changed semantic request or signed grant", i)
		}
		var response learnerapi.Response
		check(t, json.Unmarshal(w.replies[i], &response))
		if response.Grant == nil || response.Grant.Proof.Evidence.Snapshot.Payload.Generation != generation {
			t.Fatalf("did not exercise ahead-of-relay grant: %+v", response.Grant)
		}
	}
}

func TestLearnerRelayArtifactGrantRollover(t *testing.T) {
	for _, operation := range []string{"read", "create"} {
		for _, mode := range []string{"catch-up", "unchanged-then-catch-up", "failed-then-catch-up"} {
			t.Run(operation+"/"+mode, func(t *testing.T) {
				x := newLearnerRelay(t)
				local := &rolloverTrust{view: x.f.view}
				latest := advanceRolloverSigner(t, x.f)
				endpoint, wire := observeRolloverRelay(t, x)
				o := x.options[0]
				o.Endpoint, o.Trust = endpoint, local
				refreshes, wantAttempts := 0, 1
				if mode != "catch-up" {
					wantAttempts = 3
				}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				deadline, _ := ctx.Deadline()
				o.RefreshTrust = func(attempt context.Context) error {
					refreshes++
					if d, ok := attempt.Deadline(); !ok || d.After(deadline) {
						t.Error("trust refresh escaped caller deadline")
					}
					if refreshes < wantAttempts {
						if mode == "failed-then-catch-up" {
							return errors.New("authority fetch unavailable")
						}
						return nil
					}
					local.install(latest)
					return nil
				}
				c, err := clusterapp.NewLearnerClientWithPolicy(o)
				check(t, err)
				a := x.s.Assignments[0]
				if operation == "read" {
					var out bytes.Buffer
					check(t, c.Read(ctx, a, a.BaseModel, &out))
					if !bytes.Equal(out.Bytes(), x.body) {
						t.Fatal("catch-up changed artifact bytes")
					}
				} else {
					check(t, c.Write(ctx, a, run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}, bytes.NewReader(x.body)))
				}
				if refreshes != wantAttempts {
					t.Fatalf("refreshes=%d want=%d", refreshes, wantAttempts)
				}
				wire.assert(t, wantAttempts, 1, latest.Generation())
			})
		}
	}
}

func TestLearnerRelayArtifactGrantRolloverRevoked(t *testing.T) {
	x := newLearnerRelay(t)
	local := &rolloverTrust{view: x.f.view}
	peer := advanceRolloverSigner(t, x.f)
	revoked := rolloverSnapshot(t, x.f, peer.Generation()+1, []trust.Revocation{{Kind: "principal", ID: x.f.serverID, Mode: "compromise", Reason: "retired coordinator", FirstGeneration: peer.Generation() + 1}})
	endpoint, wire := observeRolloverRelay(t, x)
	o := x.options[0]
	o.Endpoint, o.Trust = endpoint, local
	refreshes := 0
	o.RefreshTrust = func(context.Context) error { refreshes++; local.install(revoked); return nil }
	c, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var out bytes.Buffer
	err = c.Read(ctx, x.s.Assignments[0], x.s.Assignments[0].BaseModel, &out)
	if err == nil || errors.Is(err, trust.ErrTrustBehind) || errors.Is(err, private.ErrUnavailable) || !strings.Contains(err.Error(), "revoked") || refreshes != 1 || out.Len() != 0 || ctx.Err() != nil {
		t.Fatalf("revoked grant not rejected permanently before transfer: refreshes=%d bytes=%d err=%v", refreshes, out.Len(), err)
	}
	wire.assert(t, 1, 0, peer.Generation())
}

func TestLearnerRelayArtifactGrantRolloverRechecksConnectionPrincipals(t *testing.T) {
	for _, revokedIdentity := range []string{"local-worker", "tls-coordinator-certificate"} {
		for _, operation := range []string{"read", "create"} {
			t.Run(revokedIdentity+"/"+operation, func(t *testing.T) {
				x := newLearnerRelay(t)
				local := &rolloverTrust{view: x.f.view}
				peer := advanceRolloverSigner(t, x.f)
				now := x.f.clock()
				check(t, trust.VerifyWorkload(x.f.a, peer, x.identities[0].Chain, x.bindings[0], now))
				tlsPrincipal, err := trust.AuthenticatePrincipal(x.f.a, peer, x.f.server.Chain, now)
				check(t, err)
				signer := x.f.server
				revocation := trust.Revocation{Kind: "principal", ID: x.bindings[0].Principal, Mode: "compromise", Reason: "connection identity revoked during catch-up", FirstGeneration: peer.Generation() + 1}
				if revokedIdentity == "tls-coordinator-certificate" {
					signer = x.f.leaf(t, trust.Coordinator, x.f.serverID)
					revocation.Kind, revocation.ID = "certificate", tlsPrincipal.Serial
				}
				latest := rolloverSnapshot(t, x.f, peer.Generation()+1, []trust.Revocation{revocation})
				signingPrincipal, err := trust.AuthenticatePrincipal(x.f.a, latest, signer.Chain, now)
				check(t, err)
				if signingPrincipal.Principal != tlsPrincipal.Principal || signingPrincipal.Role != trust.Coordinator {
					t.Fatal("grant signer changed the coordinator principal")
				}
				if revokedIdentity == "tls-coordinator-certificate" {
					if signingPrincipal.Serial == tlsPrincipal.Serial {
						t.Fatal("grant signer must use a different certificate than the TLS responder")
					}
					check(t, trust.VerifyWorkload(x.f.a, latest, x.identities[0].Chain, x.bindings[0], now))
					if _, err := trust.AuthenticatePrincipal(x.f.a, latest, x.f.server.Chain, now); err == nil {
						t.Fatal("actual TLS responder certificate was not revoked")
					}
				} else {
					if err := trust.VerifyWorkload(x.f.a, latest, x.identities[0].Chain, x.bindings[0], now); err == nil {
						t.Fatal("local worker was not revoked")
					}
				}
				node, err := trust.AuthenticatePrincipal(x.f.a, latest, x.artifactFixture.clients[0].Chain, now)
				check(t, err)
				if node.Role != trust.Node || node.Principal != x.bindings[0].Participant {
					t.Fatal("owning node is not independently authorized")
				}
				// Leave the fixture TLS policies on peer trust. Otherwise their
				// rejection could mask removal of the fresh response-principal guard.
				endpoint, wire := observeRolloverRelay(t, x, signer)
				o := x.options[0]
				o.Endpoint, o.Trust = endpoint, local
				refreshes, nodePolicies := 0, 0
				o.RefreshTrust = func(context.Context) error { refreshes++; local.install(latest); return nil }
				nodePolicy := o.NodePolicy
				o.NodePolicy = func() (*managedtls.Policy, error) { nodePolicies++; return nodePolicy() }
				c, err := clusterapp.NewLearnerClientWithPolicy(o)
				check(t, err)
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				a := x.s.Assignments[0]
				var out bytes.Buffer
				src := bytes.NewReader(x.body)
				if operation == "read" {
					err = c.Read(ctx, a, a.BaseModel, &out)
				} else {
					err = c.Write(ctx, a, run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}, src)
				}
				if err == nil || errors.Is(err, trust.ErrTrustBehind) || errors.Is(err, private.ErrUnavailable) || !strings.Contains(err.Error(), "revoked") || ctx.Err() != nil {
					t.Fatalf("revoked connection principal was not rejected permanently after valid grant catch-up: parent=%v err=%v", ctx.Err(), err)
				}
				if refreshes != 1 || nodePolicies != 0 || out.Len() != 0 || src.Len() != len(x.body) {
					t.Fatalf("revoked connection reached node-authenticated transfer: refreshes=%d nodePolicies=%d received=%d unread=%d", refreshes, nodePolicies, out.Len(), src.Len())
				}
				wire.assert(t, 1, 0, peer.Generation())
				wire.mu.Lock()
				reply := bytes.Clone(wire.replies[0])
				wire.mu.Unlock()
				var response learnerapi.Response
				check(t, json.Unmarshal(reply, &response))
				check(t, response.Grant.Validate())
				accepted, err := trust.VerifyProof(x.f.a, latest, response.Grant.Proof, response.Grant.Proof.Request, now)
				check(t, err)
				if accepted.Generation != latest.Generation() || response.Grant.Proof.Evidence.Serial != signingPrincipal.Serial {
					t.Fatal("denial was not isolated from a valid refreshed grant proof")
				}
			})
		}
	}
}

func TestLearnerRelayArtifactGrantRolloverWaitsForBudget(t *testing.T) {
	for _, mode := range []string{"nil-refresh", "unchanged-refresh", "failed-refresh"} {
		t.Run(mode, func(t *testing.T) {
			x := newLearnerRelay(t)
			local := &rolloverTrust{view: x.f.view}
			peer := advanceRolloverSigner(t, x.f)
			endpoint, wire := observeRolloverRelay(t, x)
			o := x.options[0]
			o.Endpoint, o.Trust = endpoint, local
			fetchErr := errors.New("authority offline")
			refreshes := 0
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			deadline, _ := ctx.Deadline()
			if mode != "nil-refresh" {
				o.RefreshTrust = func(attempt context.Context) error {
					refreshes++
					if d, ok := attempt.Deadline(); !ok || d.After(deadline) {
						t.Error("refresh extended retry budget")
					}
					if mode == "failed-refresh" {
						return fetchErr
					}
					return nil
				}
			}
			c, err := clusterapp.NewLearnerClientWithPolicy(o)
			check(t, err)
			var out bytes.Buffer
			err = c.Read(ctx, x.s.Assignments[0], x.s.Assignments[0].BaseModel, &out)
			if !errors.Is(err, private.ErrUnavailable) || !errors.Is(err, context.DeadlineExceeded) || ctx.Err() == nil || out.Len() != 0 {
				t.Fatalf("trust lag became permanent before budget or escaped deadline: bytes=%d err=%v", out.Len(), err)
			}
			v, err := local.Current(context.Background(), x.f.clock())
			check(t, err)
			if v.Generation() >= peer.Generation() {
				t.Fatal("relay installed peer snapshot without independent refresh")
			}
			wire.mu.Lock()
			defer wire.mu.Unlock()
			attempts := len(wire.requests)
			// A successful HTTP response can arrive as the deadline expires,
			// before client-side proof verification. Require repeated verification
			// rather than assuming every server reply was consumed.
			if attempts < 2 || (mode != "nil-refresh" && (refreshes < 2 || refreshes > attempts)) {
				t.Fatalf("did not retry while behind: attempts=%d refreshes=%d", attempts, refreshes)
			}
			wire.assertLocked(t, attempts, 0, peer.Generation())
		})
	}
}

func TestLearnerRelayArtifactGrantRolloverRefreshAttemptDeadline(t *testing.T) {
	x := newLearnerRelay(t)
	local := &rolloverTrust{view: x.f.view}
	peer := advanceRolloverSigner(t, x.f)
	endpoint, wire := observeRolloverRelay(t, x)
	o := x.options[0]
	o.Endpoint, o.Trust = endpoint, local
	// Keep the parent outside the production 55-second semantic budget so
	// only the 50-second attempt deadline can end the first refresh.
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	deadline, _ := ctx.Deadline()
	refreshes := 0
	var started time.Time
	var firstDeadline time.Time
	var firstTimeout error
	o.RefreshTrust = func(attempt context.Context) error {
		refreshes++
		d, ok := attempt.Deadline()
		if !ok || !d.Before(deadline) {
			t.Error("refresh lacks a bounded per-attempt deadline")
		}
		if refreshes == 1 {
			firstDeadline = d
			if d.Before(started.Add(50*time.Second)) || time.Until(d) > 50*time.Second {
				t.Error("first refresh does not use the 50-second attempt budget")
			}
			<-attempt.Done()
			firstTimeout = attempt.Err()
			if !errors.Is(firstTimeout, context.DeadlineExceeded) || ctx.Err() != nil {
				t.Errorf("first refresh ended without its own timeout: attempt=%v parent=%v", firstTimeout, ctx.Err())
			}
			return firstTimeout
		}
		if !d.After(firstDeadline) || d.After(firstDeadline.Add(5*time.Second)) {
			t.Error("retry reused the expired deadline or extended the original 55-second budget")
		}
		local.install(peer)
		return nil
	}
	c, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	var out bytes.Buffer
	started = time.Now()
	err = c.Read(ctx, x.s.Assignments[0], x.s.Assignments[0].BaseModel, &out)
	elapsed := time.Since(started)
	if err != nil {
		t.Fatalf("refresh attempt timeout became permanent with overall budget remaining: parent=%v refreshes=%d err=%v", ctx.Err(), refreshes, err)
	}
	if refreshes != 2 || !errors.Is(firstTimeout, context.DeadlineExceeded) || ctx.Err() != nil || elapsed < 50*time.Second || elapsed >= 55*time.Second || !bytes.Equal(out.Bytes(), x.body) {
		t.Fatalf("retry did not catch up within the original budget: refreshes=%d elapsed=%v parent=%v bytes=%d", refreshes, elapsed, ctx.Err(), out.Len())
	}
	wire.assert(t, 2, 1, peer.Generation())
}
