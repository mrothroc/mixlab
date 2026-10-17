package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
)

type closedConnectionRecoveryNode struct {
	clusterapp.LearnerRecoveryNode
	start func(context.Context, string, clusterapp.NodeLearnerRetryStart) error
}

func (n closedConnectionRecoveryNode) StartLearnerRetry(ctx context.Context, job string, q clusterapp.NodeLearnerRetryStart) error {
	return n.start(ctx, job, q)
}

func closedConnectionRecovery(t *testing.T, f *recoveryFixture, start func(context.Context, string, clusterapp.NodeLearnerRetryStart) error) clusterapp.DiLoCoRecovery {
	t.Helper()
	r := f.recovery(t)
	lookup := r.Node
	r.Node = func(a run.Allocation) (clusterapp.LearnerRecoveryNode, error) {
		n, err := lookup(a)
		return closedConnectionRecoveryNode{LearnerRecoveryNode: n, start: start}, err
	}
	return r
}

func closedRecoveryConnectionError(f *recoveryFixture, job string) error {
	return &url.Error{Op: "Put", URL: f.x.url + "/v1/agent/learners/" + job + "/retry-start", Err: &net.OpError{Op: "write", Net: "tcp", Err: net.ErrClosed}}
}

func setClosedRecoveryTime(f *recoveryFixture, unix int64) {
	f.x.f.mu.Lock()
	defer f.x.f.mu.Unlock()
	f.x.f.now = time.Unix(unix, 0)
}

func renewRecoveryLeaseThroughBudget(t *testing.T, f *recoveryFixture) {
	t.Helper()
	ctx := context.Background()
	allocation := f.x.prepare.Signed.Manifest.Slot.Allocation
	lease, err := f.node.LeaseStatus(ctx, allocation.LeaseID)
	check(t, err)
	// These tests tick recovery directly, without the service's lease keeper.
	// Replace the admission-clipped lease through the real post-handoff API.
	ttl := int(coordinator.LearnerRecoverySeconds) + 60
	renewed, err := f.node.Renew(ctx, nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: lease.ID, ExpectedVersion: lease.Version, TTLSeconds: ttl})
	check(t, err)
	if renewed.Expires <= f.x.f.clock().Unix()+coordinator.LearnerRecoverySeconds || renewed.WorkloadDeadline != lease.WorkloadDeadline || renewed.Expires > lease.WorkloadDeadline {
		t.Fatal("recovery fixture needs lease headroom within the original workload deadline")
	}
}

func TestProductionLearnerRecoveryRetryStartClosedConnection(t *testing.T) {
	for _, mode := range []string{"unapplied_bare", "unapplied_wrapped", "applied_response_lost"} {
		t.Run(mode, func(t *testing.T) {
			ctx := context.Background()
			f := newRecoveryFixture(t)
			renewRecoveryLeaseThroughBudget(t, f)
			before := f.state(t)
			f.stop(t)
			_, err := f.recovery(t).Tick(ctx, f.assignment.Binding.RunID)
			check(t, err)
			intent := *f.state(t).Recovery
			applied := mode == "applied_response_lost"
			var requests [][]byte
			start := func(ctx context.Context, job string, q clusterapp.NodeLearnerRetryStart) error {
				body, err := json.Marshal(q)
				check(t, err)
				requests = append(requests, body)
				if len(requests) == 1 {
					if applied {
						check(t, f.node.StartLearnerRetry(ctx, job, q))
					}
					if mode == "unapplied_bare" {
						return net.ErrClosed
					}
					return closedRecoveryConnectionError(f, job)
				}
				if !bytes.Equal(requests[0], body) {
					t.Fatal("retry-start replay changed signed request bytes")
				}
				return f.node.StartLearnerRetry(ctx, job, q)
			}
			_, err = closedConnectionRecovery(t, f, start).Tick(ctx, f.assignment.Binding.RunID)
			if !errors.Is(err, net.ErrClosed) || !strings.Contains(err.Error(), "start retry") {
				t.Fatal("did not exercise retry-start closed connection", err)
			}
			s := f.state(t)
			if s.Phase == coordinator.Failed || s.Recovery == nil || *s.Recovery != intent || len(s.Attempts) != 1 {
				t.Fatal("closed connection failed permanently or changed recovery intent", s.Failure)
			}
			attempt := s.Attempts[0]
			x, err := f.x.store.ActiveExecution(ctx)
			check(t, err)
			if !applied && (!x.Learner.Stopped || x.Learner.Recovery == nil || x.Workload.Generation != 1) {
				t.Fatal("unapplied request changed node hosting permission")
			}
			if applied {
				// An ambiguous response may hide a successful mutation. Prove the
				// exact captured request is idempotent at the real node HTTP API.
				var replay clusterapp.NodeLearnerRetryStart
				check(t, json.Unmarshal(requests[0], &replay))
				check(t, start(ctx, attempt.Allocation.JobID, replay))
			} else {
				// Reopen coordinator/admission state, then retry at the same signed
				// time so the complete request, including proof timestamps, is exact.
				_, err = closedConnectionRecovery(t, f, start).Tick(ctx, f.assignment.Binding.RunID)
				check(t, err)
			}
			s = f.state(t)
			if s.Recovery == nil || *s.Recovery != intent || len(s.Attempts) != 1 || s.Attempts[0] != attempt {
				t.Fatal("replay renewed the recovery budget or changed the bound attempt")
			}
			setClosedRecoveryTime(f, intent.Started+coordinator.LearnerRecoverySeconds-1)
			_, err = closedConnectionRecovery(t, f, start).Tick(ctx, f.assignment.Binding.RunID)
			check(t, err)
			s = f.state(t)
			x, err = f.x.store.ActiveExecution(ctx)
			check(t, err)
			if s.Phase != before.Phase || s.Recovery != nil || len(s.Attempts) != 1 || s.Attempts[0] != attempt || attempt.AttemptID != intent.Attempt || len(requests) != 2 {
				t.Fatal("closed-connection recovery did not finish the original attempt")
			}
			if !reflect.DeepEqual(before.Plan.Slots, s.Plan.Slots) || !reflect.DeepEqual(before.Assignments, s.Assignments) || x.Learner.Stopped || x.Learner.Recovery != nil || x.Workload.Generation != 2 || x.Learner.Assignment.AttemptID != intent.Attempt || x.Learner.Assignment.ExpectedPrivateHash != intent.PrivateHash || len(x.Learner.History) != 1 || f.x.initializations.Load() != 2 {
				t.Fatal("closed-connection replay duplicated credentials or changed allocation/private state")
			}
		})
	}
}

func TestProductionLearnerRecoveryRetryStartClosedConnectionBudget(t *testing.T) {
	ctx := context.Background()
	f := newRecoveryFixture(t)
	renewRecoveryLeaseThroughBudget(t, f)
	f.stop(t)
	_, err := f.recovery(t).Tick(ctx, f.assignment.Binding.RunID)
	check(t, err)
	intent := *f.state(t).Recovery
	var requests []clusterapp.NodeLearnerRetryStart
	start := func(ctx context.Context, job string, q clusterapp.NodeLearnerRetryStart) error {
		deadline, ok := ctx.Deadline()
		remaining := time.Duration(coordinator.LearnerRecoverySeconds-(f.x.f.clock().Unix()-intent.Started)) * time.Second
		if !ok || time.Until(deadline) > remaining {
			t.Fatal("retry-start escaped the original recovery deadline")
		}
		requests = append(requests, q)
		if len(requests) > 1 {
			first := requests[0]
			// The live signer may change diagnostic SignedAt, but never the
			// operation binding, persisted grant or authority-issued credential.
			if q.Acknowledgement.Body != first.Acknowledgement.Body || q.Acknowledgement.Coordinator != first.Acknowledgement.Coordinator || q.Acknowledgement.Proof.Request != first.Acknowledgement.Proof.Request || !reflect.DeepEqual(q.Grant, first.Grant) || !reflect.DeepEqual(q.Credential, first.Credential) {
				t.Fatal("elapsed time changed the pending retry operation")
			}
		}
		return closedRecoveryConnectionError(f, job)
	}
	// Leave ample real time for signed HTTP/issuance work on slow race builds;
	// the next tick still probes the exact persisted recovery cutoff.
	for _, elapsed := range []int64{0, coordinator.LearnerRecoverySeconds / 2} {
		setClosedRecoveryTime(f, intent.Started+elapsed)
		_, err = closedConnectionRecovery(t, f, start).Tick(ctx, f.assignment.Binding.RunID)
		if !errors.Is(err, net.ErrClosed) {
			t.Fatal("expected retryable closed connection", err)
		}
		s := f.state(t)
		if s.Phase == coordinator.Failed || s.Recovery == nil || *s.Recovery != intent || len(s.Attempts) != 1 || s.Attempts[0].AttemptID != intent.Attempt {
			t.Fatal("closed connection renewed the budget or changed attempt identity", s.Failure)
		}
	}
	setClosedRecoveryTime(f, intent.Started+coordinator.LearnerRecoverySeconds)
	_, err = closedConnectionRecovery(t, f, start).Tick(ctx, f.assignment.Binding.RunID)
	check(t, err)
	s := f.state(t)
	if s.Phase != coordinator.Failed || !strings.Contains(s.Failure, "budget exhausted") || len(requests) != 2 || len(s.Attempts) != 1 || s.Attempts[0].AttemptID != intent.Attempt {
		t.Fatal("closed connection extended the persisted recovery budget", s.Failure)
	}
	x, err := f.x.store.ActiveExecution(ctx)
	check(t, err)
	if !x.Learner.Stopped || x.Workload.Generation != 1 || x.Learner.Recovery == nil || len(x.Learner.History) != 0 || f.x.initializations.Load() != 2 {
		t.Fatal("expired retry launched a replacement or duplicated credentials")
	}
}
