package clusterapp

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
)

func admissionMaintenanceService(t *testing.T) (DiLoCoService, *admission.Saga, *admissionLeaseFixture) {
	t.Helper()
	k, f := admissionLeaseSetup(t, 2)
	f.committed(t)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	path, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	inbox, err := admission.OpenInbox(path)
	check(t, err)
	ctx := context.Background()
	check(t, inbox.Accept(ctx, f.q))
	saga, err := inbox.Resume(ctx, f.q.Launch.Run)
	check(t, err)
	ports := admission.Ports{
		Clock: func() int64 { return f.now.Load() },
		Reserve: func(_ context.Context, c recruitment.Candidate, _ nodeagent.Reserve) (nodeagent.Lease, error) {
			// The fixture prepared the registration templates above. Admission
			// still sees each original reserved lease, as on lost-ACK replay.
			return *f.nodes[c.Capabilities.Node].reserved, nil
		},
		AbortReservation: func(context.Context, recruitment.Candidate, nodeagent.Reserve) error { return nil },
		BuildPlan: func(_ context.Context, _ recruitment.LaunchPlan, _ admission.Specification, leases []nodeagent.Lease) (run.Plan, error) {
			return buildDiLoCoPlan(f.q, leases)
		},
		Stage: func(context.Context, run.Plan, coordinator.EvaluationPolicy) error { return nil },
		Prepare: func(_ context.Context, _ run.Plan, _ admission.Specification, index int) (run.Registration, error) {
			return f.state.Commit.Registrations[index], nil
		},
		Start:    func(context.Context, run.Plan, run.Registration) error { return nil },
		Register: func(context.Context, run.Registration) error { return nil },
		Commit:   func(context.Context, run.Commit) error { return nil },
		Abort:    func(context.Context, run.Plan, coordinator.EvaluationPolicy) (*run.Commit, error) { return nil, nil },
		Finalize: func(context.Context, run.Plan, run.Commit, int) error { return nil },
	}
	s := DiLoCoService{Inbox: inbox, Clock: k.Clock, Admission: func(context.Context, admission.Submission) (admission.Ports, error) { return ports, nil }, Run: func(context.Context, admission.Submission, admission.Snapshot) error {
		t.Error("round processing before Done")
		return nil
	}}
	return s, saga, f
}

func TestDiLoCoServiceMaintenanceTransientBlocksForwardEffects(t *testing.T) {
	s, saga, f := admissionMaintenanceService(t)
	ctx := context.Background()
	builds, maintains := 0, 0
	s.Admission = func(context.Context, admission.Submission) (admission.Ports, error) {
		builds++
		return admission.Ports{}, nil
	}
	s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
		maintains++
		return errors.New("temporary node outage")
	}
	check(t, s.step(ctx, f.q.Launch.Run))
	v, err := saga.Snapshot(ctx)
	check(t, err)
	if maintains != 1 || builds != 0 || v.Aborting || v.Nodes[0].Touched {
		t.Fatal("transient maintenance advanced or canceled admission")
	}
	observation, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
	check(t, err)
	if observation.Error != "temporary node outage" {
		t.Fatal("maintenance failure hidden", observation)
	}
}

func TestDiLoCoServiceMaintenanceLossAllowsCompensation(t *testing.T) {
	s, saga, f := admissionMaintenanceService(t)
	ctx := context.Background()
	maintains := 0
	s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
		maintains++
		return ErrAdmissionLeaseLost
	}
	for range 5 {
		check(t, s.step(ctx, f.q.Launch.Run))
	}
	v, err := saga.Snapshot(ctx)
	check(t, err)
	if maintains != 1 || !v.Aborting || !v.Done || v.Committed {
		t.Fatal("lease loss blocked compensation or advanced admission", v)
	}
}

func TestDiLoCoServiceMaintenanceDoesNotBlockCommittedHandoff(t *testing.T) {
	for _, maintenanceErr := range []error{errors.New("temporary node outage"), ErrAdmissionLeaseLost} {
		t.Run(maintenanceErr.Error(), func(t *testing.T) {
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
				t.Fatal("fixture did not stop at committed handoff")
			}
			maintains, finalized, runs := 0, 0, 0
			build := s.Admission
			s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
				maintains++
				return maintenanceErr
			}
			s.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
				p, err := build(ctx, q)
				p.Finalize = func(context.Context, run.Plan, run.Commit, int) error { finalized++; return nil }
				return p, err
			}
			s.Run = func(_ context.Context, _ admission.Submission, v admission.Snapshot) error {
				if !v.Done {
					t.Error("opened rounds during partial handoff")
				}
				runs++
				return nil
			}
			for range 3 {
				check(t, s.step(ctx, f.q.Launch.Run))
			}
			v, err = saga.Snapshot(ctx)
			check(t, err)
			if !v.Done || !v.Committed || finalized != 2 || maintains != 3 || runs != 0 {
				t.Fatalf("handoff stalled or rounds started early: finalized=%d maintenance=%d runs=%d", finalized, maintains, runs)
			}
			check(t, s.step(ctx, f.q.Launch.Run))
			if runs != 1 || maintains != 3 {
				t.Fatal("Done did not transfer to committed lifecycle")
			}
		})
	}
}
