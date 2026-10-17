package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/workload"
)

type rejectedRecoveryNode struct {
	status    nodeagent.LearnerStatus
	statusErr error
	retryErr  error
}

func (n rejectedRecoveryNode) LearnerStatus(context.Context, string) (nodeagent.LearnerStatus, error) {
	return n.status, n.statusErr
}
func (n rejectedRecoveryNode) AuthorizeLearnerRetry(context.Context, string, NodeLearnerRetryRequest) (NodeLearnerRetryPrepared, error) {
	return NodeLearnerRetryPrepared{}, n.retryErr
}
func (n rejectedRecoveryNode) StartLearnerRetry(context.Context, string, NodeLearnerRetryStart) error {
	return errors.New("unexpected retry start")
}

// Use the actual service step, recovery/model owners and installed event adapter.
// Only remote node/authority calls and the numerical artifact store are fakes.
func TestDiLoCoServiceRecoveryRejectionDiagnostic(t *testing.T) {
	for _, stage := range []string{"recovery learner status", "authorize retry"} {
		for _, observeFails := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/observation-fails=%t", stage, observeFails), func(t *testing.T) {
				s, saga, f := admissionMaintenanceService(t)
				ctx := context.Background()
				for range 30 {
					v, err := saga.Snapshot(ctx)
					check(t, err)
					if v.Done {
						break
					}
					check(t, s.step(ctx, f.q.Launch.Run))
				}
				v, err := saga.Snapshot(ctx)
				check(t, err)
				if !v.Done || !v.Committed {
					t.Fatal("fixture did not reach committed service ownership")
				}
				f.now.Store(1100)
				repo := &recoveryTestRepository{state: f.state}
				nodes := lifecycleLeases(f)
				models := lifecycleModels(f, repo, nodes)
				model, err := models.Model(ctx, f.q.Launch.Run)
				check(t, err)
				slot := f.plan.Slots[0]
				attempt, _, err := repo.state.CurrentAttempt(slot.Member.WorkerID)
				check(t, err)
				rejection := &NodeHTTPError{Status: http.StatusConflict}
				node := rejectedRecoveryNode{status: nodeagent.LearnerStatus{Binding: f.state.Commit.Binding, Allocation: slot.Allocation, WorkerID: slot.Member.WorkerID, Attempt: attempt, Stopped: true}}
				if stage == "authorize retry" {
					repo.state.PrivateHashes[0] = f.plan.Contract.ConfigHash
					check(t, model.BeginLearnerRecovery(ctx, f.q.Launch.Run, slot.Member.WorkerID))
					node.retryErr = rejection
				} else {
					node.statusErr = rejection
				}
				models.Recover = func(ctx context.Context, q admission.Submission, m coordinator.Model) (bool, error) {
					r := DiLoCoRecovery{Model: m, Clock: models.Clock, Node: func(run.Allocation) (LearnerRecoveryNode, error) { return node, nil },
						Current: func(context.Context) (trust.VerifiedSnapshot, error) {
							t.Fatal("rejected operation reached trust")
							return trust.VerifiedSnapshot{}, nil
						},
						Sign: func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
							t.Fatal("rejected operation reached signer")
							return trust.SignedProof{}, nil
						},
						Grant: func(context.Context, workload.SignedRequest) (workload.Grant, error) {
							t.Fatal("rejected operation reached grant")
							return workload.Grant{}, nil
						},
						Issue: func(context.Context, workload.Grant) (workload.Result, error) {
							t.Fatal("rejected operation reached authority")
							return workload.Result{}, nil
						},
					}
					return r.Tick(ctx, q.Launch.Run)
				}
				s.Run = models.Tick
				var events []error
				o := DiLoCoCoordinatorOptions{Event: func(err error) { events = append(events, err) }}
				s.Event = o.runEvent
				if observeFails {
					s.Clock = func() time.Time { return time.Unix(0, 0) }
				}
				err = s.step(ctx, f.q.Launch.Run)
				if (err != nil) != observeFails {
					t.Fatal("unexpected observation result", err)
				}
				if len(events) != 1 || !errors.Is(events[0], rejection) {
					t.Fatalf("original recovery error lost: %v", events)
				}
				for _, want := range []string{"run " + f.q.Launch.Run, stage, "HTTP 409"} {
					if !strings.Contains(events[0].Error(), want) {
						t.Fatalf("diagnostic missing %q: %v", want, events[0])
					}
				}
				if repo.state.Phase != coordinator.Failed || repo.state.Failure != "learner same-allocation recovery permanently rejected" {
					t.Fatal("rejection did not publish sanitized failure")
				}
				var public DiLoCoSubmissionStatus
				public.setRunState(repo.state)
				payload, err := json.Marshal(public)
				check(t, err)
				for _, forbidden := range []string{stage, "HTTP 409", f.plan.Contract.ConfigHash, "credential", "proof"} {
					if strings.Contains(string(payload), forbidden) {
						t.Fatalf("public status exposed diagnostic or authority: %s", payload)
					}
				}
				if strings.Contains(events[0].Error(), f.plan.Contract.ConfigHash) {
					t.Fatal("local event dumped private recovery intent")
				}
			})
		}
	}
}
