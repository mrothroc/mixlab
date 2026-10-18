//go:build darwin || linux

package clusterapp

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestLearnerPreparationRetriesExpiredEmbeddedProof(t *testing.T) {
	f := newLearnerRuntimeFixtureMode(t, false)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	r := &f.remote.runtime
	m := &f.prepare.Signed.Manifest
	m.Plan.ExpiresAt = testNow.Add(30 * time.Minute).Unix()
	var err error
	m.Plan.SHA256, err = m.Plan.Digest()
	check(t, err)
	request, err := m.SigningRequest()
	check(t, err)
	identity, signer, err := f.controller.Active(testNow)
	check(t, err)
	view, err := f.authority.Current(ctx, testNow)
	check(t, err)
	f.prepare.Signed.Proof, err = trust.SignPrincipalProof(r.Anchor, identity.Chain, signer, view, request, testNow)
	check(t, err)
	originalSigned := f.prepare.Signed
	lease, err := r.Store.LeaseStatus(ctx, f.actor, m.Slot.Allocation.LeaseID, testNow)
	check(t, err)
	lease, err = r.Store.RenewLease(ctx, f.actor, nodeagent.LeaseCommand{Lease: lease.ID, ExpectedVersion: lease.Version, IdempotencyKey: strings.Repeat("7", 32), TTLSeconds: 1800}, testNow)
	check(t, err)
	f.prepare.ExpectedLeaseVersion = lease.Version

	interrupted := errors.New("numerical preparation interrupted")
	var numerical atomic.Int32
	first := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
		numerical.Add(1)
		return workerprobe.Plan{}, interrupted
	})
	if _, err := first.PrepareLearner(ctx, f.actor, f.prepare); !errors.Is(err, interrupted) {
		t.Fatalf("first preparation did not reach numerical stage: %v", err)
	}
	x, err := r.Store.ActiveExecution(ctx)
	check(t, err)
	if x == nil || x.Lease.State != nodeagent.Prepared || x.Learner == nil || !reflect.DeepEqual(x.Learner.Signed, originalSigned) {
		t.Fatal("interruption did not preserve the generic prepared journal")
	}
	accepted := x.Learner.Accepted

	// Replace the owner only after its operation has joined, then refresh local
	// trust without replacing the original controller-signed admission proof.
	check(t, first.StopLearnerPreparation(ctx, m.Slot.Allocation.JobID))
	next := testNow.Add(trust.SnapshotLifetime + time.Second)
	view, err = f.authority.Current(ctx, next)
	check(t, err)
	snapshot, err := f.authority.Snapshots.Load(next)
	check(t, err)
	check(t, refreshPrincipal(ctx, r.Principal, snapshot, next))
	check(t, refreshPrincipal(ctx, f.controller, snapshot, next))
	identity, _, err = f.controller.Active(next)
	check(t, err)
	f.actor, err = trust.AuthenticatePrincipal(r.Anchor, view, identity.Chain, next)
	check(t, err)
	if _, err := nodejob.AcceptLearner(r.Anchor, view, f.actor, m.Slot.Allocation.NodeID, originalSigned, next); err == nil {
		t.Fatal("fixture did not expire the original embedded proof snapshot")
	}
	r.Clock = func() time.Time { return next }
	f.ports, err = NodeWorkloadPorts(r.Principal, f.root, r.Clock)
	check(t, err)
	restarted := preparationApp(t, f, context.Background(), func(context.Context, []byte) (workerprobe.Plan, error) {
		numerical.Add(1)
		return preparationNumerical(f), nil
	})
	replayed := errors.New("replayed completed generic preparation")
	restarted.options.Preparation.Probe = func(context.Context) (workerprobe.Report, error) {
		return workerprobe.Report{}, replayed
	}
	restarted.options.Preparation.DatasetID = func(context.Context, string) (string, error) {
		return "", replayed
	}
	out, err := restarted.PrepareLearner(ctx, f.actor, f.prepare)
	check(t, err)
	check(t, out.Registration.ValidateFor(m.Plan))
	again, err := restarted.PrepareLearner(ctx, f.actor, f.prepare)
	check(t, err)
	if !reflect.DeepEqual(out, again) || numerical.Load() != 2 {
		t.Fatalf("retry changed result or replayed numerical work: calls=%d", numerical.Load())
	}
	x, err = r.Store.ActiveExecution(ctx)
	check(t, err)
	if x == nil || x.Learner == nil || x.Learner.Accepted != accepted || !reflect.DeepEqual(x.Learner.Signed, originalSigned) {
		t.Fatal("historical retry replaced the original signed admission or acceptance receipt")
	}

	denied := errors.New("current caller no longer authorized")
	var callerChecks atomic.Int32
	deniedCtx := context.WithValue(ctx, nodeCallerCheckKey{}, func() error {
		callerChecks.Add(1)
		return denied
	})
	if _, err := restarted.PrepareLearner(deniedCtx, f.actor, f.prepare); !errors.Is(err, denied) || callerChecks.Load() == 0 {
		t.Fatalf("cached historical result bypassed current caller authentication: %v", err)
	}
}
