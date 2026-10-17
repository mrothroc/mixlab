package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/trust"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func receiptPendingUpdate(t *testing.T, x *relayFixture) private.Update {
	t.Helper()
	a := x.s.Assignments[0]
	hash, err := run.Digest(a)
	check(t, err)
	u := private.Update{Format: "mixlab_outer_update_v1", ID: "pending-update", AssignmentHash: hash, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: x.bindings[0].Participant, IslandID: x.s.Plan.Slots[0].Member.IslandID, Contract: a.Contract, Delta: run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: strings.Repeat("d", 64), Attempted: a.InnerSteps, Committed: a.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 1, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: x.f.clock().Unix(), FinishedAt: x.f.clock().Unix()}
	check(t, u.Validate())
	return u
}

func receiptPendingHTTP(t *testing.T, x *relayFixture, endpoint string, q learnerapi.Request) (int, []byte) {
	t.Helper()
	p, err := x.options[0].Policy()
	check(t, err)
	c, err := p.HTTPClient(5 * time.Second)
	check(t, err)
	defer c.CloseIdleConnections()
	b, err := json.Marshal(q)
	check(t, err)
	r, err := c.Post(endpoint+learnerapi.Route, "application/json", bytes.NewReader(b))
	check(t, err)
	defer func() { _ = r.Body.Close() }()
	out, err := io.ReadAll(r.Body)
	check(t, err)
	return r.StatusCode, out
}

func TestLearnerReceiptPendingHTTPAndCommittedLookup(t *testing.T) {
	x := newLearnerRelay(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	u := receiptPendingUpdate(t, x)
	q := x.pollRequest()
	q.Call.Operation, q.Call.Update = "receipt", &u
	status, body := receiptPendingHTTP(t, x, x.url, q)
	if status != http.StatusAccepted || string(body) != `{"code":"receipt_pending"}` {
		t.Fatalf("missing receipt did not return explicit pending: HTTP %d %s", status, body)
	}
	if _, err := x.clients[0].Receipt(ctx, u); !errors.Is(err, private.ErrReceiptPending) || errors.Is(err, private.ErrUnavailable) {
		t.Fatal("authenticated missing receipt lost typed result", err)
	}
	bad := q
	bad.Workload.Attempt = id(t)
	if status, _ := receiptPendingHTTP(t, x, x.url, bad); status != http.StatusForbidden {
		t.Fatal("unauthorized lookup became pending", status)
	}
	wrong := u
	wrong.ParticipantID = x.bindings[1].Participant
	bad = q
	bad.Call.Update = &wrong
	if status, _ := receiptPendingHTTP(t, x, x.url, bad); status != http.StatusConflict {
		t.Fatal("invalid update participant became pending", status)
	}
	check(t, x.clients[0].Start(ctx, private.Start{Format: "mixlab_learner_start_v1", Binding: u.Binding, WorkerID: u.WorkerID, AssignmentHash: u.AssignmentHash, PrivateHash: strings.Repeat("c", 64)}))
	check(t, x.clients[0].Write(ctx, x.s.Assignments[0], u.Delta, bytes.NewReader(x.body)))
	committed, err := x.clients[0].Submit(ctx, u)
	check(t, err)
	found, err := x.clients[0].Receipt(ctx, u)
	check(t, err)
	if found != committed {
		t.Fatal("committed receipt changed after pending lookup")
	}
	wrong = u
	wrong.Loss++
	if _, err := x.clients[0].Receipt(ctx, wrong); err == nil || errors.Is(err, private.ErrReceiptPending) || errors.Is(err, private.ErrUnavailable) {
		t.Fatal("conflicting committed receipt was hidden", err)
	}
}

type pendingReceiptRevocationRepository struct {
	coordinator.Repository
	coordinator.EvaluationReceipts
	afterLookup func()
}

func (r pendingReceiptRevocationRepository) Receipt(ctx context.Context, runID, updateID string) (private.Receipt, error) {
	out, err := r.Repository.Receipt(ctx, runID, updateID)
	r.afterLookup()
	return out, err
}

func TestLearnerReceiptPendingRechecksGatewayAuthorization(t *testing.T) {
	x := newLearnerRelay(t)
	revoked := rolloverSnapshot(t, x.f, x.f.view.Generation()+1, []trust.Revocation{{Kind: "principal", ID: x.bindings[0].Principal, Mode: "compromise", Reason: "retired worker", FirstGeneration: x.f.view.Generation() + 1}})
	x.model.Repository = pendingReceiptRevocationRepository{Repository: x.model.Repository, EvaluationReceipts: x.model.Repository.(coordinator.EvaluationReceipts), afterLookup: func() {
		x.f.mu.Lock()
		defer x.f.mu.Unlock()
		x.f.view = revoked
	}}
	server := serveCapabilities(t, x.g.Policy, retryRelayHandler(t, x))
	u := receiptPendingUpdate(t, x)
	q := x.pollRequest()
	q.Call.Operation, q.Call.Update = "receipt", &u
	if status, body := receiptPendingHTTP(t, x, server.URL, q); status != http.StatusForbidden || bytes.Contains(body, []byte("receipt_pending")) {
		t.Fatalf("revoked caller received pending instead of denial: HTTP %d %s", status, body)
	}
}

func TestLearnerReceiptPendingGatewayToIPCOnlyPendingIsQuiet(t *testing.T) {
	x := newLearnerRelay(t)
	var diagnostics []error
	s := &learneripc.Server{Remote: x.clients[0], MaxBytes: 1024, OnError: func(operation string, err error) {
		if operation != "call/receipt" {
			t.Error("lost receipt operation context", operation)
		}
		diagnostics = append(diagnostics, err)
	}}
	defer s.Close()
	u := receiptPendingUpdate(t, x)
	for _, pending := range []bool{true, false} {
		update := u
		if !pending {
			update.ParticipantID = x.bindings[1].Participant
		}
		q := x.pollRequest().Call
		q.Operation, q.Update = "receipt", &update
		payload, err := json.Marshal(struct {
			Operation string               `json:"operation"`
			Call      private.RelayRequest `json:"call"`
		}{"call", q})
		check(t, err)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		reply, err := s.Handle(ctx, wc.Envelope{Kind: wc.KindRoundRequest, PayloadKind: learneripc.PayloadKind, PayloadVersion: 1, Payload: payload})
		cancel()
		check(t, err)
		want := `{"error":"receipt not yet available","error_code":"receipt_pending"}`
		if !pending {
			want = `{"error":"operation rejected","error_code":"rejected"}`
		}
		if string(reply.Payload) != want {
			t.Fatalf("unexpected sanitized wire result: %s", reply.Payload)
		}
		if pending && len(diagnostics) != 0 {
			t.Fatal("expected missing receipt emitted error diagnostic", diagnostics)
		}
	}
	if len(diagnostics) != 1 || !strings.Contains(diagnostics[0].Error(), "HTTP 409") {
		t.Fatal("real conflict lost underlying local diagnostic", diagnostics)
	}
}

func TestLearnerReceiptPendingRechecksClientPrincipals(t *testing.T) {
	for _, identity := range []string{"local-worker", "tls-coordinator-certificate"} {
		t.Run(identity, func(t *testing.T) {
			x := newLearnerRelay(t)
			local := &rolloverTrust{view: x.f.view}
			revocation := trust.Revocation{Kind: "principal", ID: x.bindings[0].Principal, Mode: "compromise", Reason: "receipt response identity revoked", FirstGeneration: x.f.view.Generation() + 1}
			if identity == "tls-coordinator-certificate" {
				peer, err := trust.AuthenticatePrincipal(x.f.a, x.f.view, x.f.server.Chain, x.f.clock())
				check(t, err)
				revocation.Kind, revocation.ID = "certificate", peer.Serial
			}
			latest := rolloverSnapshot(t, x.f, revocation.FirstGeneration, []trust.Revocation{revocation})
			var calls atomic.Int32
			server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				// Keep TLS policy trust unchanged so only the fresh response guard
				// can reject the newly revoked identity.
				local.install(latest)
				w.WriteHeader(http.StatusAccepted)
				_, _ = io.WriteString(w, `{"code":"receipt_pending"}`)
			}))
			o := x.options[0]
			o.Trust, o.Endpoint = local, server.URL
			c, err := clusterapp.NewLearnerClientWithPolicy(o)
			check(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, err = c.Receipt(ctx, receiptPendingUpdate(t, x))
			if err == nil || errors.Is(err, private.ErrReceiptPending) || errors.Is(err, private.ErrUnavailable) || !strings.Contains(err.Error(), "revoked") || calls.Load() != 1 || ctx.Err() != nil {
				t.Fatalf("revoked connection became quiet pending or retried: calls=%d err=%v", calls.Load(), err)
			}
		})
	}
}

func TestLearnerReceiptPendingClientRequiresExplicitAuthenticatedResult(t *testing.T) {
	x := newLearnerRelay(t)
	u := receiptPendingUpdate(t, x)
	for _, tc := range []struct {
		name      string
		status    int
		body      string
		operation string
		pending   bool
	}{
		{"pending", 202, `{"code":"receipt_pending"}`, "receipt", true},
		{"409-is-not-pending", 409, `{"code":"receipt_pending"}`, "receipt", false},
		{"403-is-not-pending", 403, `{"code":"receipt_pending"}`, "receipt", false},
		{"200-is-not-pending", 200, `{"code":"receipt_pending"}`, "receipt", false},
		{"missing-code", 202, `{}`, "receipt", false},
		{"wrong-code", 202, `{"code":"rejected"}`, "receipt", false},
		{"extra-payload", 202, `{"code":"receipt_pending","receipt":{}}`, "receipt", false},
		{"submit-cannot-be-pending", 202, `{"code":"receipt_pending"}`, "submit", false},
		{"poll-cannot-be-pending", 202, `{"code":"receipt_pending"}`, "poll", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			c := retryRelayClient(t, x, server.URL, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var err error
			switch tc.operation {
			case "receipt":
				_, err = c.Receipt(ctx, u)
			case "submit":
				_, err = c.Submit(ctx, u)
			case "poll":
				_, err = c.Poll(ctx)
			}
			if err == nil || errors.Is(err, private.ErrReceiptPending) != tc.pending || errors.Is(err, private.ErrUnavailable) || calls.Load() != 1 || ctx.Err() != nil {
				t.Fatalf("ambiguous result was retried or hidden: calls=%d err=%v", calls.Load(), err)
			}
		})
	}
	t.Run("untrusted-responder", func(t *testing.T) {
		foreign, err := learnerapi.NewPolicy(x.f.leaf(t, trust.Coordinator, id(t)), x.f.a, artifactTrust{x.f}, x.f.clock)
		check(t, err)
		var calls atomic.Int32
		server := serveCapabilities(t, foreign, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			calls.Add(1)
			w.WriteHeader(http.StatusAccepted)
			_, _ = io.WriteString(w, `{"code":"receipt_pending"}`)
		}))
		_, err = retryRelayClient(t, x, server.URL, nil).Receipt(context.Background(), u)
		if err == nil || errors.Is(err, private.ErrReceiptPending) || calls.Load() != 0 {
			t.Fatal("unauthenticated pending result was accepted", calls.Load(), err)
		}
	})
}
