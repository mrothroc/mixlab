package nodeleases

import (
	"context"
	"errors"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

func TestKeeperReleaseConflictReconcilesOnlyExactCleanup(t *testing.T) {
	statusFailure := errors.New("status unavailable")
	for _, phase := range []coordinator.Phase{coordinator.Complete, coordinator.Failed} {
		for _, tc := range []struct {
			name   string
			change func(*nodeagent.Lease)
			err    error
		}{
			{name: "released"},
			{name: "still_running", change: func(l *nodeagent.Lease) { l.State = nodeagent.Running }},
			{name: "still_draining", change: func(l *nodeagent.Lease) { l.State = nodeagent.Releasing }},
			{name: "same_version", change: func(l *nodeagent.Lease) { l.Version-- }},
			{name: "older_version", change: func(l *nodeagent.Lease) { l.Version -= 2 }},
			{name: "zero_version", change: func(l *nodeagent.Lease) { l.Version = 0 }},
			{name: "foreign_lease", change: func(l *nodeagent.Lease) { l.ID = "foreign" }},
			{name: "foreign_node", change: func(l *nodeagent.Lease) { l.Node = "foreign" }},
			{name: "foreign_job", change: func(l *nodeagent.Lease) { l.Job = "foreign" }},
			{name: "foreign_run", change: func(l *nodeagent.Lease) { l.Run = "foreign" }},
			{name: "foreign_coordinator", change: func(l *nodeagent.Lease) { l.Coordinator = "foreign" }},
			{name: "foreign_commit", change: func(l *nodeagent.Lease) { l.RunCommitHash = "foreign" }},
			{name: "foreign_kind", change: func(l *nodeagent.Lease) { l.JobKind = "trainer" }},
			{name: "status_error", err: statusFailure},
			{name: "canceled", err: context.Canceled},
		} {
			t.Run(string(phase)+"/"+tc.name, func(t *testing.T) {
				k, f := setup(t)
				f.s.Phase = phase
				slot := f.s.Plan.Slots[0]
				reads, releases := 0, 0
				conflict := errors.Join(ErrReleaseConflict, errors.New("HTTP 409"))
				k.Nodes = controlledNodes{Nodes: f,
					status: func(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
						reads++
						l, err := f.Status(ctx, a)
						if reads == 2 {
							if tc.change != nil {
								tc.change(&l)
							}
							err = tc.err
						}
						return l, err
					},
					release: func(_ context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
						releases++
						l := f.leases[a.LeaseID]
						l.State, l.Version = nodeagent.Released, l.Version+1
						l.CompletedAttempt, l.CleanupReason = q.CompletedAttempt, "completed"
						f.leases[a.LeaseID] = l
						// The error response itself cannot prove physical cleanup.
						return l, conflict
					},
				}
				got := k.maintainSlot(context.Background(), f.s, slot, true)
				if reads != 2 || releases != 1 || got.lost {
					t.Fatalf("unexpected reconciliation: reads=%d releases=%d result=%+v", reads, releases, got)
				}
				if tc.name == "released" {
					if got.err != nil || !got.released {
						t.Fatalf("confirmed cleanup not accepted: %+v", got)
					}
					got = k.maintainSlot(context.Background(), f.s, slot, true)
					if got.err != nil || !got.released || releases != 1 || reads != 3 {
						t.Fatalf("repeated cleanup issued another release: %+v", got)
					}
				} else if got.released || !errors.Is(got.err, conflict) || (tc.err != nil && !errors.Is(got.err, tc.err)) {
					t.Fatalf("unconfirmed cleanup or error hidden: %+v", got)
				}
			})
		}
	}
}

func TestKeeperReleaseConflictPreservesCompletionAttempt(t *testing.T) {
	for _, change := range []func(*nodeagent.Lease){
		func(l *nodeagent.Lease) { l.CompletedAttempt = "foreign" },
		func(l *nodeagent.Lease) { l.CompletedAttempt = "" },
		func(l *nodeagent.Lease) { l.CleanupReason = "released" },
		func(l *nodeagent.Lease) { l.CleanupReason = "expired" },
	} {
		k, f := setup(t)
		f.s.Phase = coordinator.Complete
		slot := f.s.Plan.Slots[0]
		k.Nodes = controlledNodes{Nodes: f, release: func(_ context.Context, a run.Allocation, q nodeagent.LeaseCommand) (nodeagent.Lease, error) {
			l := f.leases[a.LeaseID]
			l.State, l.Version = nodeagent.Released, l.Version+1
			l.CompletedAttempt, l.CleanupReason = q.CompletedAttempt, "completed"
			change(&l)
			f.leases[a.LeaseID] = l
			return l, ErrReleaseConflict
		}}
		got := k.maintainSlot(context.Background(), f.s, slot, true)
		if got.released || !errors.Is(got.err, ErrReleaseConflict) {
			t.Fatalf("different completion accepted: %+v", got)
		}
	}
}

func TestKeeperReleaseDoesNotReconcileOtherErrors(t *testing.T) {
	for _, failure := range []error{errors.New("denied"), context.Canceled, context.DeadlineExceeded} {
		k, f := setup(t)
		f.s.Phase = coordinator.Failed
		reads := 0
		k.Nodes = controlledNodes{Nodes: f,
			status: func(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
				reads++
				return f.Status(ctx, a)
			},
			release: func(context.Context, run.Allocation, nodeagent.LeaseCommand) (nodeagent.Lease, error) {
				return nodeagent.Lease{}, failure
			},
		}
		got := k.maintainSlot(context.Background(), f.s, f.s.Plan.Slots[0], true)
		if reads != 1 || got.released || !errors.Is(got.err, failure) {
			t.Fatalf("non-conflict error reconciled: reads=%d result=%+v", reads, got)
		}
	}
}
