package integration

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/trust"
)

type contentionLearnerRepository struct {
	coordinator.Repository
	coordinator.EvaluationReceipts
	once    sync.Once
	entered chan struct{}
	release chan struct{}
	result  chan error
}

func (r *contentionLearnerRepository) Save(ctx context.Context, id string, revision uint64, change coordinator.Change) error {
	first := false
	r.once.Do(func() {
		first = true
		close(r.entered)
		select {
		case <-r.release:
		case <-ctx.Done():
		}
	})
	err := r.Repository.Save(ctx, id, revision, change)
	if first {
		r.result <- err
	}
	return err
}

type contentionLearnerResponse struct {
	*httptest.ResponseRecorder
	wire http.ResponseWriter
}

func (w contentionLearnerResponse) Unwrap() http.ResponseWriter { return w.wire }

func contentionLearnerHandler(t *testing.T, x *relayFixture, repo coordinator.Repository) http.Handler {
	t.Helper()
	grants, err := learnerapi.OpenGrants(artifactPath(t))
	check(t, err)
	m := x.model
	m.Repository = repo
	h, err := (learnerapi.Gateway{Grants: grants, Policy: x.g.Policy, Anchor: x.f.a, Trust: artifactTrust{x.f}, Coordinator: x.s.Plan.Coordinator, Model: m, Clock: x.f.clock, Timeout: 5 * time.Second, Sign: func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(x.f.a, x.f.server.Chain, x.f.server.Key, x.f.view, q, x.f.now)
	}}).Handler()
	check(t, err)
	return h
}

// The gateway and recovery actor write the same real durable run head. Hold
// the learner's Save until recovery has advanced that head, forcing a lost CAS
// rather than injecting a generic conflict error at the HTTP boundary.
func TestLearnerGatewayTLSRecoveryContention(t *testing.T) {
	for _, operation := range []string{"start", "submit"} {
		for _, transition := range []string{"begin", "finish"} {
			t.Run(operation+"/"+transition, func(t *testing.T) {
				x := newLearnerRelay(t)
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				start := func(index int) private.Start {
					a := x.s.Assignments[index]
					h, err := run.Digest(a)
					check(t, err)
					return private.Start{Format: "mixlab_learner_start_v1", Binding: a.Binding, WorkerID: a.WorkerID, AssignmentHash: h, PrivateHash: strings.Repeat("c", 64)}
				}
				check(t, x.clients[1].Start(ctx, start(1)))
				q := x.pollRequest()
				q.Call.Operation = operation
				firstStart := start(0)
				if operation == "start" {
					q.Call.Start = &firstStart
				} else {
					check(t, x.clients[0].Start(ctx, firstStart))
					a := x.s.Assignments[0]
					delta := run.Blob{Kind: "delta_f32", Ref: artifactRef(x.body)}
					check(t, x.clients[0].Write(ctx, a, delta, bytes.NewReader(x.body)))
					q.Call.Update = &private.Update{Format: "mixlab_outer_update_v1", ID: "contention-update", AssignmentHash: firstStart.AssignmentHash, Binding: a.Binding, Round: a.Round, BaseRound: a.Round, BaseModelHash: a.BaseModel.Ref.SHA256, WorkerID: a.WorkerID, ParticipantID: x.bindings[0].Participant, IslandID: x.s.Plan.Slots[0].Member.IslandID, Contract: a.Contract, Delta: delta, DType: "float32", TensorCount: 1, PrecisionHash: a.PrecisionHash, PrivateHash: strings.Repeat("d", 64), Attempted: a.InnerSteps, Committed: a.InnerSteps, EffectiveTokens: 8, EffectiveExamples: 1, DeltaL2: 1, DeltaMaxAbs: 1, StartedAt: x.f.now.Unix(), FinishedAt: x.f.now.Unix()}
				}
				runID, recovering := q.Call.Binding.RunID, x.bindings[1].Worker
				if transition == "finish" {
					check(t, x.model.BeginLearnerRecovery(ctx, runID, recovering))
				}
				before, err := x.model.Repository.Load(ctx, runID)
				check(t, err)
				repo := &contentionLearnerRepository{Repository: x.model.Repository, EvaluationReceipts: x.model.Repository.(coordinator.EvaluationReceipts), entered: make(chan struct{}), release: make(chan struct{}), result: make(chan error, 1)}
				h := contentionLearnerHandler(t, x, repo)
				var mu sync.Mutex
				var statuses []int
				var requests [][]byte
				server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Error(err)
						return
					}
					r.Body = io.NopCloser(bytes.NewReader(body))
					recorder := httptest.NewRecorder()
					h.ServeHTTP(contentionLearnerResponse{recorder, w}, r)
					mu.Lock()
					statuses = append(statuses, recorder.Code)
					requests = append(requests, body)
					mu.Unlock()
					w.Header().Set("Content-Type", recorder.Header().Get("Content-Type"))
					w.WriteHeader(recorder.Code)
					_, _ = w.Write(recorder.Body.Bytes())
				}))
				o := x.options[0]
				o.Endpoint = server.URL
				client, err := clusterapp.NewLearnerClientWithPolicy(o)
				check(t, err)
				done := make(chan error, 1)
				go func() { _, err := client.Call(ctx, q.Call); done <- err }()
				select {
				case <-repo.entered:
				case err := <-done:
					t.Fatalf("request did not reach Save: %v", err)
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
				if transition == "begin" {
					err = x.model.BeginLearnerRecovery(ctx, runID, recovering)
				} else {
					err = x.model.FinishLearnerRecovery(ctx, runID, before.Recovery.Attempt)
				}
				close(repo.release)
				check(t, err)
				if err = <-repo.result; !errors.Is(err, coordinator.ErrConcurrent) || !errors.Is(err, coordinator.ErrConflict) {
					t.Fatalf("did not reproduce real repository contention: %v", err)
				}
				if err = <-done; err != nil {
					t.Fatalf("unaffected learner terminated on recovery CAS: %v", err)
				}
				mu.Lock()
				gotStatuses := append([]int(nil), statuses...)
				identical := len(requests) == 2 && bytes.Equal(requests[0], requests[1])
				mu.Unlock()
				if !reflect.DeepEqual(gotStatuses, []int{http.StatusServiceUnavailable, http.StatusOK}) || !identical {
					t.Fatalf("expected exact semantic retry after 503, got %v identical=%v", gotStatuses, identical)
				}
				after, err := x.model.Repository.Load(ctx, runID)
				check(t, err)
				attempt, generation, err := after.CurrentAttempt(x.bindings[0].Worker)
				check(t, err)
				if after.Revision != before.Revision+2 || attempt != x.bindings[0].Attempt || generation != x.bindings[0].Generation || len(after.Attempts) != len(before.Attempts) {
					t.Fatal("contention changed unaffected attempt or committed more than once")
				}
				_, err = client.Call(ctx, q.Call)
				check(t, err)
				replayed, err := x.model.Repository.Load(ctx, runID)
				check(t, err)
				if replayed.Revision != after.Revision {
					t.Fatal("exact duplicate changed durable revision")
				}
				if q.Call.Start != nil {
					q.Call.Start.PrivateHash = strings.Repeat("e", 64)
				} else {
					q.Call.Update.PrivateHash = strings.Repeat("e", 64)
				}
				if _, err = client.Call(ctx, q.Call); err == nil {
					t.Fatal("changed immutable replay input accepted")
				}
				mu.Lock()
				defer mu.Unlock()
				if !reflect.DeepEqual(statuses, []int{503, 200, 200, 409}) {
					t.Fatalf("semantic conflict retried: %v", statuses)
				}
			})
		}
	}
}
