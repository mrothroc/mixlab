package clusterapp

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
)

func TestDiLoCoServiceLoadedPrecommitMaintenance(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, saga, f := admissionMaintenanceService(t)
		ctx := context.Background()
		check(t, s.step(ctx, f.q.Launch.Run)) // Persist the original reserve intent.
		start := time.Now()
		s.Maintain = func(ctx context.Context, _ admission.Submission, _ admission.Snapshot) error {
			deadline, ok := ctx.Deadline()
			if !ok || deadline.Sub(start) != diLoCoServiceBudget {
				t.Fatal("precommit service truncated maintenance", deadline)
			}
			time.Sleep(40 * time.Second)
			return ctx.Err()
		}
		build := s.Admission
		reserved := 0
		s.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
			p, err := build(ctx, q)
			reserve := p.Reserve
			p.Reserve = func(ctx context.Context, c recruitment.Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
				time.Sleep(20 * time.Second)
				if err := ctx.Err(); err != nil {
					return nodeagent.Lease{}, err
				}
				reserved++
				return reserve(ctx, c, q)
			}
			return p, err
		}
		check(t, s.step(ctx, f.q.Launch.Run))
		v, err := saga.Snapshot(ctx)
		check(t, err)
		if reserved != 1 || v.Nodes[0].Lease == nil || v.Aborting || time.Since(start) != time.Minute {
			t.Fatal("loaded precommit maintenance truncated or replaced original reservation")
		}
	})
}

func TestDiLoCoServiceMaintenanceExhaustionLeavesHandoffBudget(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, saga, f := admissionMaintenanceService(t)
		ctx := context.Background()
		for range 30 {
			v, err := saga.Snapshot(ctx)
			check(t, err)
			if v.Committed {
				break
			}
			check(t, s.step(ctx, f.q.Launch.Run))
		}
		v, err := saga.Snapshot(ctx)
		check(t, err)
		if !v.Committed || v.Done {
			t.Fatal("fixture missed handoff boundary")
		}
		s.Maintain = func(ctx context.Context, _ admission.Submission, _ admission.Snapshot) error {
			ctx, cancel := context.WithTimeout(ctx, admissionLeaseBudget)
			defer cancel()
			<-ctx.Done()
			return ctx.Err()
		}
		build := s.Admission
		finalized := 0
		s.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) != learnerAttemptBudget {
				t.Fatal("maintenance consumed reconciliation headroom", time.Until(deadline))
			}
			p, err := build(ctx, q)
			p.Finalize = func(ctx context.Context, _ run.Plan, _ run.Commit, _ int) error {
				time.Sleep(20 * time.Second)
				if err := ctx.Err(); err != nil {
					return err
				}
				finalized++
				return nil
			}
			return p, err
		}
		start := time.Now()
		check(t, s.step(ctx, f.q.Launch.Run))
		v, err = saga.Snapshot(ctx)
		check(t, err)
		if finalized != 1 || !v.Nodes[0].Finalized || v.Done || time.Since(start) != admissionLeaseBudget+20*time.Second {
			t.Fatal("maintenance timeout blocked handoff or opened rounds")
		}
	})
}

func TestDiLoCoServiceMaintenanceCrossingAdmissionDeadlineFencesEffects(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, saga, f := admissionMaintenanceService(t)
		ctx := context.Background()
		check(t, s.step(ctx, f.q.Launch.Run))
		start := time.Now()
		s.Clock = func() time.Time { return time.Unix(f.q.Deadline-10, 0).Add(time.Since(start)) }
		s.Maintain = func(ctx context.Context, _ admission.Submission, _ admission.Snapshot) error {
			// Even a port returning success after crossing admission expiry must
			// not enable a forward effect under the longer service envelope.
			time.Sleep(40 * time.Second)
			f.now.Store(s.Clock().Unix())
			return ctx.Err()
		}
		build := s.Admission
		reserved := 0
		s.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
			p, err := build(ctx, q)
			p.Reserve = func(context.Context, recruitment.Candidate, nodeagent.Reserve) (nodeagent.Lease, error) {
				reserved++
				return nodeagent.Lease{}, errors.New("expired forward effect")
			}
			return p, err
		}
		check(t, s.step(ctx, f.q.Launch.Run))
		v, err := saga.Snapshot(ctx)
		check(t, err)
		if reserved != 0 || !v.Aborting || v.Committed || v.Nodes[0].Lease != nil || v.Deadline != f.q.Deadline {
			t.Fatal("longer service budget extended immutable admission")
		}
	})
}

func TestDiLoCoServiceLoadedMaintenanceCancellationDoesNotCompensate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, saga, f := admissionMaintenanceService(t)
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		s.Maintain = func(ctx context.Context, _ admission.Submission, _ admission.Snapshot) error {
			<-ctx.Done()
			return ctx.Err()
		}
		s.Admission = func(context.Context, admission.Submission) (admission.Ports, error) {
			t.Error("shutdown reached admission effects")
			return admission.Ports{}, nil
		}
		if err := s.step(ctx, f.q.Launch.Run); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal("service lost parent cancellation", err)
		}
		v, err := saga.Snapshot(context.Background())
		check(t, err)
		if v.Aborting || v.Nodes[0].Touched {
			t.Fatal("shutdown changed admission authority")
		}
	})
}
