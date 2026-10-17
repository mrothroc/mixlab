package integration

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

type nodePreflightRecovery struct {
	clusterapp.LearnerRecoveryNode
	start func(context.Context, string, clusterapp.NodeLearnerRetryStart) error
}

func TestNodeRetryRequestDeadlinePreservesRecoveryAndCatchesUp(t *testing.T) {
	const (
		nodeBudget    = 50 * time.Second
		clientBudget  = nodeBudget + 5*time.Second
		refreshBudget = 15 * time.Second
	)
	var f *recoveryFixture
	var refreshes atomic.Int32
	type refreshDeadline struct {
		remaining time.Duration
		err       error
	}
	expired := make(chan refreshDeadline, 1)
	n := &recoveryNodeTrust{}
	n.refresh = func(ctx context.Context) error {
		if refreshes.Add(1) == 1 {
			deadline, ok := ctx.Deadline()
			if !ok {
				return errors.New("node refresh has no deadline")
			}
			remaining := time.Until(deadline)
			// With the 50s route and 55s client budgets, the inner 15s trust
			// refresh deadline expires first. Exercise that real deadline,
			// not a shortened test timeout or synthetic refresh error.
			<-ctx.Done()
			expired <- refreshDeadline{remaining: remaining, err: ctx.Err()}
			return ctx.Err()
		}
		n.pin(f.x.f.view)
		return nil
	}
	f = newRecoveryFixtureWithNodeTrust(t, strings.Repeat("a", 64), true, n)
	f.stop(t)
	n.pin(f.x.f.view)
	advanceRecoverySnapshot(t, f)
	ctx, cancel := context.WithTimeout(context.Background(), clientBudget+5*time.Second)
	defer cancel()
	runID := f.assignment.Binding.RunID
	_, err := f.recovery(t).Tick(ctx, runID)
	check(t, err)
	begun := f.state(t)
	if begun.Recovery == nil {
		t.Fatal("recovery intent not created")
	}
	intent := *begun.Recovery
	var beforeStart *nodeagent.LocalExecution
	r := f.recovery(t)
	r.Node = func(run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
		return nodePreflightRecovery{LearnerRecoveryNode: f.node, start: func(ctx context.Context, job string, q clusterapp.NodeLearnerRetryStart) error {
			var err error
			beforeStart, err = f.x.store.ActiveExecution(ctx)
			check(t, err)
			err = f.node.StartLearnerRetry(ctx, job, q)
			if ctx.Err() != nil {
				t.Fatalf("coordinator budget expired instead of node request: %v", ctx.Err())
			}
			return err
		}}, nil
	}
	_, err = r.Tick(ctx, runID)
	var remote *clusterapp.NodeHTTPError
	if !errors.As(err, &remote) || remote.Status != http.StatusServiceUnavailable {
		t.Fatalf("node refresh expiry returned %v, want HTTP 503", err)
	}
	select {
	case result := <-expired:
		if !errors.Is(result.err, context.DeadlineExceeded) || result.remaining < refreshBudget-time.Second || result.remaining > refreshBudget {
			t.Fatalf("refresh did not exhaust actual 15s deadline within 50s node/55s client budgets: %+v", result)
		}
	default:
		t.Fatal("node returned before refresh exhausted its deadline")
	}
	waiting := f.state(t)
	afterStart, err := f.x.store.ActiveExecution(ctx)
	check(t, err)
	if waiting.Phase == coordinator.Failed || waiting.Recovery == nil || *waiting.Recovery != intent || len(waiting.Attempts) != 1 || waiting.Attempts[0].AttemptID != intent.Attempt || beforeStart == nil || !reflect.DeepEqual(beforeStart, afterStart) || !afterStart.Learner.Stopped || refreshes.Load() != 1 {
		t.Fatal("request timeout failed permanently, changed intent or mutated pending node retry")
	}
	// Charge the exhausted refresh budget to the same recovery before retry.
	f.x.f.mu.Lock()
	f.x.f.now = time.Unix(intent.Started+int64(refreshBudget/time.Second), 0)
	f.x.f.mu.Unlock()
	for range 3 {
		_, err = f.recovery(t).Tick(ctx, runID)
		check(t, err)
		s := f.state(t)
		if s.Phase == coordinator.Failed || s.Recovery != nil && *s.Recovery != intent {
			t.Fatal("catch-up replaced the original recovery intent or budget")
		}
		if s.Recovery == nil {
			x, err := f.x.store.ActiveExecution(ctx)
			check(t, err)
			if refreshes.Load() != 2 || len(s.Attempts) != 1 || s.Attempts[0].AttemptID != intent.Attempt || !reflect.DeepEqual(begun.Assignments, s.Assignments) || !reflect.DeepEqual(begun.Plan.Slots, s.Plan.Slots) || x.Learner.Stopped || x.Learner.Assignment.AttemptID != intent.Attempt || x.Learner.Recovery != nil || x.Workload.Generation != 2 || f.x.initializations.Load() != 2 {
				t.Fatal("catch-up changed allocation/attempt or recreated workload keys")
			}
			return
		}
	}
	t.Fatal("recovery did not finish after node trust caught up")
}

func (n nodePreflightRecovery) StartLearnerRetry(ctx context.Context, id string, q clusterapp.NodeLearnerRetryStart) error {
	return n.start(ctx, id, q)
}

func TestNodeRetryRejectsUnrelatedEvidenceBeforeRefresh(t *testing.T) {
	for _, mode := range []string{"job", "pending_attempt"} {
		t.Run(mode, func(t *testing.T) {
			refreshes := 0
			n := &recoveryNodeTrust{refresh: func(context.Context) error { refreshes++; return nil }}
			f := newRecoveryFixtureWithNodeTrust(t, strings.Repeat("a", 64), true, n)
			f.stop(t)
			n.pin(f.x.f.view)
			advanceRecoverySnapshot(t, f)
			checked := false
			for range 4 {
				r := f.recovery(t)
				r.Node = func(run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
					return nodePreflightRecovery{LearnerRecoveryNode: f.node, start: func(ctx context.Context, job string, q clusterapp.NodeLearnerRetryStart) error {
						checked = true
						before, err := f.x.store.ActiveExecution(ctx)
						check(t, err)
						if mode == "job" {
							job = id(t)
						} else {
							q.Acknowledgement.Body.AttemptID = id(t)
							request := q.Acknowledgement.Proof.Request
							request.Digest, err = run.Digest(q.Acknowledgement.Body)
							check(t, err)
							q.Acknowledgement.Proof, err = trust.SignPrincipalProof(f.x.f.a, f.x.coordinator.Chain, f.x.coordinator.Key, f.x.f.view, request, f.x.f.now)
							check(t, err)
						}
						check(t, q.Acknowledgement.Validate())
						_, err = trust.VerifyProof(f.x.f.a, f.x.f.view, q.Acknowledgement.Proof, q.Acknowledgement.Proof.Request, f.x.f.now)
						check(t, err)
						err = f.node.StartLearnerRetry(ctx, job, q)
						var remote *clusterapp.NodeHTTPError
						if !errors.As(err, &remote) || remote.Status != http.StatusConflict || refreshes != 0 {
							t.Fatalf("unrelated retry evidence: err=%v refreshes=%d", err, refreshes)
						}
						after, readErr := f.x.store.ActiveExecution(ctx)
						check(t, readErr)
						if !reflect.DeepEqual(before, after) {
							t.Fatal("unrelated retry evidence mutated the pending attempt")
						}
						return err
					}}, nil
				}
				_, err := r.Tick(context.Background(), f.assignment.Binding.RunID)
				if checked {
					return
				}
				check(t, err)
			}
			t.Fatal("retry start not reached")
		})
	}
}
