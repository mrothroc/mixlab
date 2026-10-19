package integration

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/trust"
)

type completeRaceRepository struct {
	coordinator.Repository
	coordinator.EvaluationReceipts
	loads        atomic.Int32
	gateAt       int32
	read, resume chan struct{}
}

func (r *completeRaceRepository) Load(ctx context.Context, id string) (coordinator.State, error) {
	s, err := r.Repository.Load(ctx, id)
	if r.loads.Add(1) == r.gateAt {
		close(r.read)
		select {
		case <-r.resume:
		case <-ctx.Done():
			err = ctx.Err()
		}
	}
	return s, err
}

type completeRaceAllocation struct {
	coordinator.Allocations
	deny bool
}

func (a completeRaceAllocation) Ready(ctx context.Context, slot run.Slot) error {
	if a.deny {
		return fmt.Errorf("live allocation unavailable")
	}
	return a.Allocations.Ready(ctx, slot)
}

// Hold either authenticated poll state read while COMPLETE commits and
// releases allocation readiness. The saved snapshot must not turn this into
// HTTP 403, nor leak its old ROUND_OPEN assignment in the final response.
func TestLearnerPollCompleteRace(t *testing.T) {
	for _, mode := range []string{"complete", "complete-final-read", "live-denied", "failed", "revoked", "foreign-attempt"} {
		t.Run(mode, func(t *testing.T) {
			x := newLearnerRelay(t)
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			r := &completeRaceRepository{Repository: x.model.Repository, EvaluationReceipts: x.model.Repository.(coordinator.EvaluationReceipts), gateAt: 1, read: make(chan struct{}), resume: make(chan struct{})}
			if mode == "complete-final-read" {
				r.gateAt = 2
			}
			defer func() {
				select {
				case <-r.resume:
				default:
					close(r.resume)
				}
			}()
			m := x.model
			m.Repository = r
			m.Allocations = completeRaceAllocation{m.Allocations, mode == "live-denied"}
			grants, err := learnerapi.OpenGrants(artifactPath(t))
			check(t, err)
			h, err := (learnerapi.Gateway{Grants: grants, Policy: x.g.Policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: x.s.Plan.Coordinator, Model: m, Clock: x.f.clock, Timeout: 10 * time.Second, Sign: func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
				return trust.SignedProof{}, fmt.Errorf("poll cannot sign grants")
			}}).Handler()
			check(t, err)
			x.url = serveCapabilities(t, x.g.Policy, h).URL
			q := x.pollRequest()
			if mode == "foreign-attempt" {
				q.Workload.Attempt = id(t)
			}
			type result struct {
				code int
				body []byte
			}
			done := make(chan result, 1)
			go func() { code, body := receiptPendingHTTP(t, x, x.url, q); done <- result{code, body} }()
			select {
			case <-r.read:
			case <-ctx.Done():
				t.Fatal("poll never reached state read")
			}
			if mode != "live-denied" {
				s, err := x.model.Repository.Load(ctx, q.Call.Binding.RunID)
				check(t, err)
				rev := s.Revision
				s.Revision++
				s.Phase, s.Round = coordinator.Complete, 1
				for i := range s.PrivateHashes {
					s.PrivateHashes[i] = strings.Repeat("a", 64)
				}
				if mode == "failed" {
					s.Phase, s.Failure = coordinator.Failed, "failed run"
				}
				check(t, x.model.Repository.Save(ctx, q.Call.Binding.RunID, rev, coordinator.Change{Next: s}))
			}
			if mode == "revoked" {
				x.f.mu.Lock()
				x.f.snapshot.Payload.Revocations = append(x.f.snapshot.Payload.Revocations, trust.Revocation{Kind: "principal", ID: x.bindings[0].Principal, Mode: "compromise", Reason: "test", FirstGeneration: x.f.view.Generation() + 1})
				x.f.snapshot.Payload.Generation++
				x.f.mu.Unlock()
				x.f.signSnapshot(t)
			}
			close(r.resume)
			select {
			case result := <-done:
				if strings.HasPrefix(mode, "complete") {
					if result.code != http.StatusOK || !strings.Contains(string(result.body), `"done":true`) || strings.Contains(string(result.body), `"assignment":{`) {
						t.Fatalf("completion lost: %d %s", result.code, result.body)
					}
					options := x.options[0]
					options.Endpoint = x.url
					client, err := clusterapp.NewLearnerClientWithPolicy(options)
					check(t, err)
					reply, err := client.Poll(ctx)
					check(t, err)
					if !reply.Done {
						t.Fatal("terminal poll retry lost completion")
					}
				} else if result.code == http.StatusOK {
					t.Fatalf("%s bypassed authorization: %s", mode, result.body)
				}
			case <-ctx.Done():
				t.Fatal("poll did not finish")
			}
		})
	}
}
