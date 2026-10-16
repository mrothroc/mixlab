package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/trust"
)

type shutdownLearnerRepository struct {
	coordinator.Repository
	coordinator.EvaluationReceipts
	loads int
	at    int
	hook  func(context.Context) error
}

func (s *shutdownLearnerRepository) Load(ctx context.Context, id string) (coordinator.State, error) {
	state, err := s.Repository.Load(ctx, id)
	s.loads++
	if err == nil && s.loads == s.at {
		err = s.hook(ctx)
	}
	return state, err
}

type shutdownLearnerTrust struct {
	artifactTrust
	hook func(context.Context) error
}

func (s shutdownLearnerTrust) Current(ctx context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
	if err := s.hook(ctx); err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	return s.artifactTrust.Current(ctx, now)
}

// Cancel the server BaseContext while an authenticated request is in flight,
// as the installed coordinator does before draining HTTP on shutdown. The
// client's context stays live so the actual HTTP status can be inspected.
func TestLearnerGatewayTLSShutdownStatus(t *testing.T) {
	x := newLearnerRelay(t)
	for _, phase := range []string{"model", "load", "authorize", "dispatch", "revalidate"} {
		for _, outcome := range []string{"canceled", "masked-cancellation", "ignored-cancellation", "live-policy", "live-context-error"} {
			t.Run(phase+"/"+outcome, func(t *testing.T) {
				serverCtx, cancel := context.WithCancel(context.Background())
				defer cancel()
				entered := make(chan struct{}, 1)
				hook := func(ctx context.Context) error {
					entered <- struct{}{}
					switch outcome {
					case "live-policy":
						return fmt.Errorf("private policy failure")
					case "live-context-error":
						return fmt.Errorf("private dependency: %w", context.Canceled)
					}
					<-ctx.Done()
					if outcome == "ignored-cancellation" {
						return nil
					}
					if outcome == "masked-cancellation" {
						return fmt.Errorf("private dependency unavailable")
					}
					return ctx.Err()
				}
				grants, err := learnerapi.OpenGrants(artifactPath(t))
				check(t, err)
				g := learnerapi.Gateway{Grants: grants, Policy: x.g.Policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: x.s.Plan.Coordinator, Model: x.model, Clock: x.f.clock, Timeout: 5 * time.Second, Sign: func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
					return trust.SignPrincipalProof(x.f.a, x.f.server.Chain, x.f.server.Key, x.f.view, q, x.f.now)
				}}
				q := x.pollRequest()
				switch phase {
				case "model":
					g.ModelForRun = func(ctx context.Context, _ string) (coordinator.Model, error) { return x.model, hook(ctx) }
				case "load", "revalidate":
					at := 1
					if phase == "revalidate" {
						at = 2
					}
					g.Model.Repository = &shutdownLearnerRepository{Repository: x.model.Repository, EvaluationReceipts: x.model.Repository.(coordinator.EvaluationReceipts), at: at, hook: hook}
				case "authorize":
					g.Trust = shutdownLearnerTrust{artifactTrust{x.f}, hook}
				case "dispatch":
					a := x.s.Assignments[0]
					q.Call.Operation = "artifact"
					q.Call.Artifact = &private.ArtifactRequest{Assignment: a, Blob: a.BaseModel, Operation: "read"}
					sign := g.Sign
					g.Sign = func(ctx context.Context, q trust.SignRequest) (trust.SignedProof, error) {
						if err := hook(ctx); err != nil {
							return trust.SignedProof{}, err
						}
						return sign(ctx, q)
					}
				}
				h, err := g.Handler()
				check(t, err)
				server := httptest.NewUnstartedServer(h)
				server.Config.BaseContext = func(net.Listener) context.Context { return serverCtx }
				server.Config.ErrorLog = log.New(io.Discard, "", 0)
				server.TLS = x.g.Policy.ServerConfig()
				server.StartTLS()
				defer server.Close()
				policy, err := x.options[0].Policy()
				check(t, err)
				client, err := policy.HTTPClient(8 * time.Second)
				check(t, err)
				defer client.CloseIdleConnections()
				body, err := json.Marshal(q)
				check(t, err)
				type result struct {
					status int
					body   string
					err    error
				}
				done := make(chan result, 1)
				go func() {
					r, err := client.Post(server.URL+learnerapi.Route, "application/json", bytes.NewReader(body))
					if err != nil {
						done <- result{err: err}
						return
					}
					defer func() { _ = r.Body.Close() }()
					b, err := io.ReadAll(r.Body)
					done <- result{r.StatusCode, string(b), err}
				}()
				select {
				case <-entered:
				case <-time.After(10 * time.Second):
					t.Fatal("request never reached test boundary")
				}
				wantStatus, wantBody := http.StatusServiceUnavailable, "learner service unavailable\n"
				if outcome == "live-policy" || outcome == "live-context-error" {
					wantStatus = http.StatusForbidden
					wantBody = map[string]string{"model": "run is not available\n", "load": "learner access rejected\n", "authorize": "learner access rejected\n", "revalidate": "learner access changed\n", "dispatch": "learner operation rejected\n"}[phase]
					if phase == "dispatch" {
						wantStatus = http.StatusConflict
					}
				} else {
					cancel()
					drain, stop := context.WithTimeout(context.Background(), 5*time.Second)
					err := server.Config.Shutdown(drain)
					stop()
					check(t, err)
				}
				r := <-done
				check(t, r.err)
				if r.status != wantStatus || r.body != wantBody {
					t.Fatalf("got %d %q, want %d %q", r.status, r.body, wantStatus, wantBody)
				}
			})
		}
	}
}
