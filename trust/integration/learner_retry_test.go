package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

func retryRelayHandler(t *testing.T, x *relayFixture) http.Handler {
	t.Helper()
	grants, err := learnerapi.OpenGrants(artifactPath(t))
	check(t, err)
	f := x.f
	h, err := (learnerapi.Gateway{Grants: grants, Policy: x.g.Policy, Anchor: f.a, Trust: artifactTrust{f}, Coordinator: x.s.Plan.Coordinator, Model: x.model, Clock: f.clock, Timeout: 10 * time.Second, Sign: func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return trust.SignPrincipalProof(f.a, f.server.Chain, f.server.Key, f.view, q, f.now)
	}}).Handler()
	check(t, err)
	return h
}

func retryRelayClient(t *testing.T, x *relayFixture, endpoint string, policy func() (*managedtls.Policy, error)) *clusterapp.LearnerClient {
	t.Helper()
	o := x.options[0]
	o.Endpoint = endpoint
	if policy != nil {
		o.Policy = policy
	}
	c, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	return c
}

func dropLearnerResponse(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	c, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		t.Error(err)
		return
	}
	if err := c.Close(); err != nil {
		t.Error(err)
	}
}

type capturedLearnerResponse struct {
	*httptest.ResponseRecorder
	wire http.ResponseWriter
}

func (w capturedLearnerResponse) Unwrap() http.ResponseWriter { return w.wire }

func TestLearnerSemanticRetryTLSCoordinatorRestart(t *testing.T) {
	x := newLearnerRelay(t)
	h := retryRelayHandler(t, x)
	server := serveCapabilities(t, x.g.Policy, h)
	address, endpoint := server.Listener.Addr().String(), server.URL
	server.Close()
	var policies atomic.Int32
	retried := make(chan struct{})
	c := retryRelayClient(t, x, endpoint, func() (*managedtls.Policy, error) {
		if policies.Add(1) == 2 {
			close(retried)
		}
		return managedtls.New(x.identities[0], x.f.verify(trust.Coordinator, x.f.serverID), x.f.clock)
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	type result struct {
		reply private.RelayResponse
		err   error
	}
	done := make(chan result, 1)
	go func() { r, err := c.Poll(ctx); done <- result{r, err} }()
	select {
	case <-retried:
	case r := <-done:
		t.Fatalf("did not survive listener outage: %v", r.err)
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	listener, err := net.Listen("tcp", address)
	check(t, err)
	restarted := httptest.NewUnstartedServer(retryRelayHandler(t, x))
	check(t, restarted.Listener.Close())
	restarted.Listener = listener
	restarted.TLS = x.g.Policy.ServerConfig()
	restarted.Config.ErrorLog = log.New(io.Discard, "", 0)
	restarted.StartTLS()
	defer restarted.Close()
	r := <-done
	check(t, r.err)
	if r.reply.Assignment == nil || *r.reply.Assignment != x.s.Assignments[0] || policies.Load() < 2 {
		t.Fatalf("restart changed assignment or reused old policy: %+v policies=%d", r.reply, policies.Load())
	}
}

func TestLearnerSemanticRetryTLSServiceDraining(t *testing.T) {
	x := newLearnerRelay(t)
	h := retryRelayHandler(t, x)
	var calls atomic.Int32
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) < 3 {
			http.Error(w, "coordinator stopping", http.StatusServiceUnavailable)
			return
		}
		h.ServeHTTP(w, r)
	}))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r, err := retryRelayClient(t, x, server.URL, nil).Poll(ctx)
	check(t, err)
	if calls.Load() != 3 || r.Assignment == nil || *r.Assignment != x.s.Assignments[0] {
		t.Fatalf("drain interrupted poll or changed assignment: %+v calls=%d", r, calls.Load())
	}
}

func TestLearnerSemanticRetryTLSUnavailableExhaustion(t *testing.T) {
	x := newLearnerRelay(t)
	var calls atomic.Int32
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, "coordinator stopping", http.StatusServiceUnavailable)
	}))
	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()
	_, err := retryRelayClient(t, x, server.URL, nil).Poll(ctx)
	if !errors.Is(err, private.ErrUnavailable) || !errors.Is(err, context.DeadlineExceeded) || calls.Load() < 1 {
		t.Fatalf("unbounded or unclassified outage: %v calls=%d", err, calls.Load())
	}
}

func TestLearnerSemanticRetryTLSLostSubmitAcknowledgment(t *testing.T) {
	x := newLearnerRelay(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	a := x.s.Assignments[0]
	hash, err := run.Digest(a)
	check(t, err)
	check(t, x.clients[0].Start(ctx, private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: hash, PrivateHash: strings.Repeat("c", 64)}))
	delta := run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}
	check(t, x.clients[0].Write(ctx, a, delta, bytes.NewReader(x.body)))
	u := private.Update{Format: "mixlab_outer_update_v1", ID: "retry-update", AssignmentHash: hash, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: x.bindings[0].Participant, IslandID: x.s.Plan.Slots[0].Member.IslandID, Contract: a.Contract, Delta: delta, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: strings.Repeat("d", 64), Attempted: a.InnerSteps, Committed: a.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 1, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: x.f.now.Unix(), FinishedAt: x.f.now.Unix()}
	before, err := x.model.Repository.Load(ctx, a.Binding.RunID)
	check(t, err)
	h := retryRelayHandler(t, x)
	var calls, policies atomic.Int32
	var mu sync.Mutex
	var requests, replies [][]byte
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(b))
		recorder := httptest.NewRecorder()
		h.ServeHTTP(capturedLearnerResponse{recorder, w}, r)
		mu.Lock()
		requests = append(requests, bytes.Clone(b))
		replies = append(replies, bytes.Clone(recorder.Body.Bytes()))
		mu.Unlock()
		if recorder.Code != http.StatusOK {
			t.Errorf("submit rejected: %d %s", recorder.Code, recorder.Body)
		}
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Length", strconv.Itoa(recorder.Body.Len()))
			_, _ = w.Write(recorder.Body.Bytes()[:5])
			w.(http.Flusher).Flush()
			dropLearnerResponse(t, w)
			return
		}
		w.WriteHeader(recorder.Code)
		_, _ = w.Write(recorder.Body.Bytes())
	}))
	c := retryRelayClient(t, x, server.URL, func() (*managedtls.Policy, error) { policies.Add(1); return x.options[0].Policy() })
	receipt, err := c.Submit(ctx, u)
	check(t, err)
	lookup, err := x.clients[0].Receipt(ctx, u)
	check(t, err)
	after, err := x.model.Repository.Load(ctx, a.Binding.RunID)
	check(t, err)
	mu.Lock()
	defer mu.Unlock()
	if calls.Load() != 2 || policies.Load() != 2 || len(requests) != 2 || !bytes.Equal(requests[0], requests[1]) || !bytes.Equal(replies[0], replies[1]) || receipt != lookup || after.Revision != before.Revision+1 {
		t.Fatalf("lost ACK changed request, receipt or commit: calls=%d policies=%d revisions=%d/%d", calls.Load(), policies.Load(), before.Revision, after.Revision)
	}
}

func TestLearnerSemanticRetryTLSRejectsNonTransportFailures(t *testing.T) {
	x := newLearnerRelay(t)
	for _, status := range []int{400, 401, 403, 409, 500, 200} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls, policies atomic.Int32
			server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				w.WriteHeader(status)
				_, _ = w.Write([]byte(`{"malformed":`))
			}))
			c := retryRelayClient(t, x, server.URL, func() (*managedtls.Policy, error) { policies.Add(1); return x.options[0].Policy() })
			_, err := c.Poll(context.Background())
			if err == nil || calls.Load() != 1 || policies.Load() != 1 {
				t.Fatal(err, calls.Load(), policies.Load())
			}
		})
	}
	t.Run("scope", func(t *testing.T) {
		var calls atomic.Int32
		server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			a := x.s.Assignments[0]
			a.WorkerID = x.bindings[1].Worker
			_ = json.NewEncoder(w).Encode(learnerapi.Response{Reply: private.RelayResponse{Format: "mixlab_learner_reply_v1", Assignment: &a}})
		}))
		_, err := retryRelayClient(t, x, server.URL, nil).Poll(context.Background())
		if err == nil || calls.Load() != 1 {
			t.Fatal(err, calls.Load())
		}
	})
	t.Run("fresh-policy-fails-after-disconnect", func(t *testing.T) {
		var calls, policies atomic.Int32
		server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1); dropLearnerResponse(t, w) }))
		c := retryRelayClient(t, x, server.URL, func() (*managedtls.Policy, error) {
			if policies.Add(1) > 1 {
				return nil, context.DeadlineExceeded
			}
			return x.options[0].Policy()
		})
		_, err := c.Poll(context.Background())
		if !errors.Is(err, context.DeadlineExceeded) || calls.Load() != 1 || policies.Load() != 2 {
			t.Fatal(err, calls.Load(), policies.Load())
		}
	})
	t.Run("certificate", func(t *testing.T) {
		var calls, policies atomic.Int32
		foreign, err := learnerapi.NewPolicy(x.f.leaf(t, trust.Coordinator, id(t)), x.f.a, artifactTrust{x.f}, x.f.clock)
		check(t, err)
		server := serveCapabilities(t, foreign, http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls.Add(1) }))
		c := retryRelayClient(t, x, server.URL, func() (*managedtls.Policy, error) {
			policies.Add(1)
			return x.options[0].Policy()
		})
		_, err = c.Poll(context.Background())
		if err == nil || calls.Load() != 0 || policies.Load() != 1 {
			t.Fatal(err, calls.Load(), policies.Load())
		}
	})
}

func TestLearnerSemanticRetryTLSAttemptTimeout(t *testing.T) {
	x := newLearnerRelay(t)
	h := retryRelayHandler(t, x)
	var calls atomic.Int32
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			_, _ = io.Copy(io.Discard, r.Body)
			<-r.Context().Done()
			return
		}
		h.ServeHTTP(w, r)
	}))
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	r, err := retryRelayClient(t, x, server.URL, nil).Poll(ctx)
	check(t, err)
	if r.Assignment == nil || calls.Load() != 2 {
		t.Fatal(r, calls.Load())
	}
}

func TestLearnerSemanticRetryDoesNotReplayArtifactStreams(t *testing.T) {
	x := newLearnerRelay(t)
	h := retryRelayHandler(t, x)
	for _, operation := range []string{"read", "write"} {
		t.Run(operation, func(t *testing.T) {
			var transfers atomic.Int32
			server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == learnerapi.Route {
					h.ServeHTTP(w, r)
					return
				}
				transfers.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				dropLearnerResponse(t, w)
			}))
			c := retryRelayClient(t, x, server.URL, nil)
			a := x.s.Assignments[0]
			var err error
			if operation == "read" {
				err = c.Read(context.Background(), a, a.BaseModel, &bytes.Buffer{})
			} else {
				err = c.Write(context.Background(), a, run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}, bytes.NewReader(x.body))
			}
			if err == nil || transfers.Load() != 1 {
				t.Fatal(err, transfers.Load())
			}
		})
	}
}
