package clusterapp

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
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
	for _, codes := range [][2]int{{503, 403}, {403, 503}, {503, 409}} {
		if recoveryTransient(errors.Join(&NodeHTTPError{Status: codes[0]}, &NodeHTTPError{Status: codes[1]})) {
			t.Fatal("mixed peer failures hid a permanent rejection", codes)
		}
	}
	if !recoveryTransient(errors.Join(&NodeHTTPError{Status: 503}, context.DeadlineExceeded)) {
		t.Fatal("independent transient peer failures became terminal")
	}
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

func TestRecoveryTransientPreservesWrappersAroundJoinedCauses(t *testing.T) {
	transient := errors.Join(io.EOF, context.DeadlineExceeded)
	remote := &net.OpError{Op: "remote error", Err: transient}
	verification := &tls.CertificateVerificationError{Err: transient}
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"certificate", verification, false},
		{"remote", remote, false},
		{"url-certificate", &url.Error{Op: "Put", URL: "https://node.invalid", Err: verification}, false},
		{"nested-remote", &net.OpError{Op: "read", Err: remote}, false},
		{"joined-certificate-first", errors.Join(verification, transient), false},
		{"joined-certificate-last", errors.Join(transient, verification), false},
		{"joined-remote-last", errors.Join(transient, remote), false},
		{"peer-rejected", fmt.Errorf("%w: %w", managedtls.ErrPeerRejected, transient), false},
		{"canceled", fmt.Errorf("%w: %w", context.Canceled, transient), false},
		{"tls-alert", errors.Join(transient, tls.AlertError(42)), false},
		{"unknown-authority", errors.Join(transient, x509.UnknownAuthorityError{}), false},
		{"invalid-certificate", errors.Join(transient, x509.CertificateInvalidError{}), false},
		{"hostname", errors.Join(transient, x509.HostnameError{}), false},
		{"wrapped-permanent-http", fmt.Errorf("maintenance: %w", errors.Join(transient, &NodeHTTPError{Status: http.StatusForbidden})), false},
		{"nested-permanent-authority", errors.Join(transient, fmt.Errorf("authority: %w", &enrollmentHTTPError{Status: http.StatusUnauthorized})), false},
		{"all-transient", transient, true},
		{"wrapped-transient", fmt.Errorf("maintenance: %w", transient), true},
		{"transport-transient", &net.OpError{Op: "read", Err: transient}, true},
		{"nested-transient-http", errors.Join(&NodeHTTPError{Status: http.StatusServiceUnavailable}, fmt.Errorf("authority: %w", errors.Join(&enrollmentHTTPError{Status: http.StatusBadGateway}, transient))), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := recoveryTransient(tc.err); got != tc.want {
				t.Fatalf("recoveryTransient(%v)=%v, want %v", tc.err, got, tc.want)
			}
		})
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
	leases map[run.Allocation]nodeagent.Lease
	renew  bool
	err    error
}

func (n outageLeaseNodes) Status(_ context.Context, allocation run.Allocation) (nodeagent.Lease, error) {
	if !n.renew {
		return nodeagent.Lease{}, n.err
	}
	lease, ok := n.leases[allocation]
	if !ok {
		return nodeagent.Lease{}, fmt.Errorf("unknown outage fixture allocation")
	}
	return lease, nil
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
					f.now.Store(1100)
					leases := make(map[run.Allocation]nodeagent.Lease, len(f.plan.Slots))
					for index, slot := range f.plan.Slots {
						f.handedOff(index, true)
						lease := f.nodes[slot.Allocation.NodeID].lease
						lease.RenewBy = 1000
						// Exercise outage exhaustion before any lease expires.
						lease.Expires = f.now.Load() + coordinator.LearnerRecoverySeconds + 300
						leases[slot.Allocation] = lease
					}
					repo := &recoveryTestRepository{state: f.state}
					artifacts := coordinatorlocal.Artifacts{}
					clock := serviceClock{func() time.Time { return time.Unix(f.now.Load(), 0) }}
					model := coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: remoteEvaluation{}}
					keeper := nodeleases.Keeper{Repository: repo, Nodes: outageLeaseNodes{leases: leases, renew: renew, err: context.DeadlineExceeded}, Clock: clock, RunID: f.q.Launch.Run, TTLSeconds: 300, Fail: model.Fail}
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
			f.now.Store(1100)
			leases := make(map[run.Allocation]nodeagent.Lease, len(f.plan.Slots))
			for index, slot := range f.plan.Slots {
				f.handedOff(index, true)
				lease := f.nodes[slot.Allocation.NodeID].lease
				lease.RenewBy = 1000
				leases[slot.Allocation] = lease
			}
			repo := &recoveryTestRepository{state: f.state}
			artifacts := coordinatorlocal.Artifacts{}
			clock := serviceClock{func() time.Time { return time.Unix(f.now.Load(), 0) }}
			model := coordinator.Model{Repository: repo, Artifacts: artifacts, Clock: clock, Evaluator: remoteEvaluation{}}
			keeper := nodeleases.Keeper{Repository: repo, Nodes: outageLeaseNodes{leases: leases, renew: renew, err: &NodeHTTPError{Status: http.StatusForbidden}}, Clock: clock, RunID: f.q.Launch.Run, TTLSeconds: 300, Fail: model.Fail}
			model.Allocations = &keeper
			models := DiLoCoModels{Inbox: &admission.Inbox{}, Principal: &principal.Store{}, Repository: repo, Artifacts: artifacts, Clock: clock.now, Recover: func(context.Context, admission.Submission, coordinator.Model) (bool, error) {
				t.Fatal("rejected lease reached learner polling")
				return false, nil
			}}
			models.cache.Store(f.q.Launch.Run, composedRun{model, keeper})
			err := models.Tick(context.Background(), f.q, admission.Snapshot{})
			if err == nil {
				t.Fatal("lease rejection not reported")
			}
			var peer *NodeHTTPError
			if !errors.As(err, &peer) || peer.Status != http.StatusForbidden || !strings.Contains(err.Error(), allocationFailureReason(f.q, clock.now())) {
				t.Fatalf("local diagnostic lost peer rejection or immutable deadline: %v", err)
			}
			if repo.state.Phase != coordinator.Failed || repo.state.Failure != allocationFailureReason(f.q, clock.now()) {
				t.Fatalf("lease failure misattributed: phase=%s reason=%q", repo.state.Phase, repo.state.Failure)
			}
		})
	}
}
