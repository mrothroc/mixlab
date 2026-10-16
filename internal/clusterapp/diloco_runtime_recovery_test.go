package clusterapp

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	"github.com/mrothroc/mixlab/coordinator/nodeleases"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestRecoveryTransientKeepsAuthenticationTerminal(t *testing.T) {
	for _, err := range []error{managedtls.ErrPeerRejected, errors.Join(managedtls.ErrPeerRejected, io.EOF), tls.AlertError(42), &net.OpError{Op: "remote error", Err: io.EOF}, &NodeHTTPError{Status: http.StatusForbidden}, &NodeHTTPError{Status: http.StatusConflict}, context.Canceled} {
		if recoveryTransient(err) {
			t.Fatalf("retried terminal error %v", err)
		}
	}
	for _, err := range []error{io.ErrUnexpectedEOF, context.DeadlineExceeded, syscall.EHOSTUNREACH, &net.OpError{Op: "dial", Err: syscall.ECONNREFUSED}, &NodeHTTPError{Status: http.StatusServiceUnavailable}} {
		if !recoveryTransient(err) {
			t.Fatalf("rejected transient error %v", err)
		}
	}
}

type recoveryTestRepository struct{ state coordinator.State }

func (r *recoveryTestRepository) Load(context.Context, string) (coordinator.State, error) {
	return r.state, nil
}
func (r *recoveryTestRepository) Receipt(context.Context, string, string) (private.Receipt, error) {
	return private.Receipt{}, coordinator.ErrNotFound
}
func (r *recoveryTestRepository) Save(ctx context.Context, _ string, revision uint64, c coordinator.Change) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if revision != r.state.Revision {
		return coordinator.ErrConflict
	}
	if err := c.Next.Validate(); err != nil {
		return err
	}
	r.state = c.Next
	return nil
}

type outageLeaseNodes struct {
	lease nodeagent.Lease
	renew bool
	err   error
}

func (n outageLeaseNodes) Status(context.Context, run.Allocation) (nodeagent.Lease, error) {
	if !n.renew {
		return nodeagent.Lease{}, n.err
	}
	return n.lease, nil
}
func (n outageLeaseNodes) Renew(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return nodeagent.Lease{}, n.err
}
func (n outageLeaseNodes) Release(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error) {
	return nodeagent.Lease{}, n.err
}

func TestDiLoCoModelsBoundsKeeperOutageBeforeRecoveryPolling(t *testing.T) {
	for _, renew := range []bool{false, true} {
		for _, pending := range []bool{false, true} {
			for _, expiredCall := range []bool{false, true} {
				t.Run(fmt.Sprintf("renew=%v/pending=%v/expired=%v", renew, pending, expiredCall), func(t *testing.T) {
					_, f := admissionLeaseSetup(t, 2)
					f.committed(t)
					f.handedOff(0, true)
					f.now.Store(1100)
					repo := &recoveryTestRepository{state: f.state}
					artifacts := coordinatorlocal.Artifacts{}
					clock := serviceClock{func() time.Time { return time.Unix(f.now.Load(), 0) }}
					model := coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: remoteEvaluation{}}
					lease := f.nodes[f.plan.Slots[0].Allocation.NodeID].lease
					lease.RenewBy = 1000
					keeper := nodeleases.Keeper{Repository: repo, Nodes: outageLeaseNodes{lease: lease, renew: renew, err: context.DeadlineExceeded}, Clock: clock, RunID: f.q.Launch.Run, TTLSeconds: 300, Fail: model.Fail}
					model.Allocations = &keeper
					if pending {
						repo.state.PrivateHashes[0] = f.plan.Contract.ConfigHash
						check(t, model.BeginLearnerRecovery(context.Background(), f.q.Launch.Run, f.plan.Slots[0].Member.WorkerID))
					}
					tick := func() {
						models := DiLoCoModels{Inbox: &admission.Inbox{}, Principal: &principal.Store{}, Repository: repo, Artifacts: artifacts, Clock: clock.now, Recover: func(context.Context, admission.Submission, coordinator.Model) (bool, error) {
							t.Fatal("lease outage reached learner polling")
							return false, nil
						}}
						models.cache.Store(f.q.Launch.Run, composedRun{model, keeper})
						ctx := context.Background()
						if expiredCall {
							var cancel context.CancelFunc
							ctx, cancel = context.WithDeadline(ctx, time.Now().Add(-time.Second))
							defer cancel()
						}
						if err := models.Tick(ctx, f.q, admission.Snapshot{}); err == nil {
							t.Fatal("outage not reported")
						}
					}
					tick()
					if !pending && repo.state.LearnerUnavailableAt != 1100 {
						t.Fatal("outage observation not persisted")
					}
					f.now.Add(coordinator.LearnerRecoverySeconds)
					tick()
					if repo.state.Phase != coordinator.Failed || repo.state.Failure == "" {
						t.Fatal("keeper bypassed recovery deadline", repo.state.Phase)
					}
				})
			}
		}
	}
}

func TestDiLoCoModelsReportsPermanentLeaseFailure(t *testing.T) {
	for _, renew := range []bool{false, true} {
		t.Run(fmt.Sprintf("renew=%v", renew), func(t *testing.T) {
			_, f := admissionLeaseSetup(t, 2)
			f.committed(t)
			f.handedOff(0, true)
			f.now.Store(1100)
			repo := &recoveryTestRepository{state: f.state}
			artifacts := coordinatorlocal.Artifacts{}
			clock := serviceClock{func() time.Time { return time.Unix(f.now.Load(), 0) }}
			model := coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: remoteEvaluation{}}
			lease := f.nodes[f.plan.Slots[0].Allocation.NodeID].lease
			lease.RenewBy = 1000
			keeper := nodeleases.Keeper{Repository: repo, Nodes: outageLeaseNodes{lease: lease, renew: renew, err: &NodeHTTPError{Status: http.StatusForbidden}}, Clock: clock, RunID: f.q.Launch.Run, TTLSeconds: 300, Fail: model.Fail}
			model.Allocations = &keeper
			models := DiLoCoModels{Inbox: &admission.Inbox{}, Principal: &principal.Store{}, Repository: repo, Artifacts: artifacts, Clock: clock.now, Recover: func(context.Context, admission.Submission, coordinator.Model) (bool, error) {
				t.Fatal("rejected lease reached learner polling")
				return false, nil
			}}
			models.cache.Store(f.q.Launch.Run, composedRun{model, keeper})
			if err := models.Tick(context.Background(), f.q, admission.Snapshot{}); err == nil {
				t.Fatal("lease rejection not reported")
			}
			if repo.state.Phase != coordinator.Failed || repo.state.Failure != "allocation lease maintenance permanently rejected" {
				t.Fatalf("lease failure misattributed: phase=%s reason=%q", repo.state.Phase, repo.state.Failure)
			}
		})
	}
}
