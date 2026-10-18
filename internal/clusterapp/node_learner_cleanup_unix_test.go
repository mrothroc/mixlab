//go:build darwin || linux

package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost"
)

func TestNodeLearnerCleanupTimeoutKeepsExecutionOwner(t *testing.T) {
	for _, phase := range []string{"stop_preparation", "destroy_credentials"} {
		t.Run(phase, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			s := f.remote.runtime.Store
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			v, err := f.authority.Current(ctx, testNow)
			check(t, err)
			m := f.prepare.Signed.Manifest
			accepted, err := nodejob.AcceptLearner(f.authority.Anchor, v, f.actor, m.Slot.Allocation.NodeID, f.prepare.Signed, testNow)
			check(t, err)
			_, err = s.PrepareLearner(ctx, f.actor, accepted, f.prepare.ExpectedLeaseVersion, testNow)
			check(t, err)
			_, err = s.PrepareLearnerWorkload(ctx, f.actor, m.Slot.Allocation.JobID, f.ports)
			check(t, err)
			var attempts atomic.Int32
			var stopped atomic.Bool
			var ready atomic.Bool
			failTwice := func() error {
				switch attempts.Add(1) {
				case 1:
					return context.DeadlineExceeded
				case 2:
					return context.Canceled
				}
				return nil
			}
			ports := NodeExecutionPorts{
				Open: func(context.Context, string, string) (*workerhost.AttemptStore, workerhost.Runner, error) {
					return nil, nil, fmt.Errorf("unstarted compensation must not open hosting")
				},
				Authorize: func(context.Context, nodeagent.LocalExecution) error {
					return fmt.Errorf("releasing job must not require new authority")
				},
				Clock: func() time.Time { return testNow }, Ready: func() { ready.Store(true) },
				StopPreparation: func(ctx context.Context, id string) error {
					if id != m.Slot.Allocation.JobID {
						return fmt.Errorf("stopped another preparation")
					}
					if phase == "stop_preparation" {
						if err := failTwice(); err != nil {
							return err
						}
					}
					stopped.Store(true)
					return nil
				},
				Cleanup: func(ctx context.Context, q nodeagent.CleanupRequest) error {
					if !stopped.Load() {
						return fmt.Errorf("cleanup preceded async preparation join")
					}
					if phase == "destroy_credentials" {
						if err := failTwice(); err != nil {
							return err
						}
					}
					return DestroyNodeWorkload(ctx, f.root, f.authority.Anchor, q)
				},
			}
			done := make(chan error, 1)
			go func() { done <- RunNodeExecution(ctx, s, ports) }()
			ticker := time.NewTicker(10 * time.Millisecond)
			defer ticker.Stop()
			for {
				select {
				case err := <-done:
					t.Fatalf("cleanup timeout terminated owner: %v", err)
				case <-ctx.Done():
					t.Fatal("cleanup did not finish", <-done)
				case <-ticker.C:
					available, err := s.Availability(testNow)
					check(t, err)
					if available.Available {
						cancel()
						check(t, <-done)
						if attempts.Load() != 3 || !ready.Load() {
							t.Fatal("cleanup did not retry through readiness")
						}
						return
					}
				}
			}
		})
	}
}

func TestNodeLearnerPartialCredentialCleanup(t *testing.T) {
	for _, stage := range []string{"before_initialize", "initialize_lost_ack", "request_lost_ack"} {
		t.Run(stage, func(t *testing.T) {
			f := newLearnerRuntimeFixtureMode(t, false)
			s := f.remote.runtime.Store
			ctx := context.Background()
			v, err := f.authority.Current(ctx, testNow)
			check(t, err)
			m := f.prepare.Signed.Manifest
			accepted, err := nodejob.AcceptLearner(f.authority.Anchor, v, f.actor, m.Slot.Allocation.NodeID, f.prepare.Signed, testNow)
			check(t, err)
			_, err = s.PrepareLearner(ctx, f.actor, accepted, f.prepare.ExpectedLeaseVersion, testNow)
			check(t, err)
			p := f.ports
			p.Initialize = func(ctx context.Context, scope workload.Scope) error {
				if stage == "before_initialize" {
					return context.DeadlineExceeded
				}
				if err := f.ports.Initialize(ctx, scope); err != nil {
					return err
				}
				if stage == "initialize_lost_ack" {
					return context.DeadlineExceeded
				}
				return nil
			}
			p.Request = func(ctx context.Context, scope workload.Scope) (workload.SignedRequest, error) {
				_, err := f.ports.Request(ctx, scope)
				if err != nil {
					return workload.SignedRequest{}, err
				}
				return workload.SignedRequest{}, context.DeadlineExceeded
			}
			_, err = s.PrepareLearnerWorkload(ctx, f.actor, m.Slot.Allocation.JobID, p)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatal(err)
			}
			check(t, s.InterruptActive(ctx, "agent_shutdown"))
			calls := 0
			check(t, s.CleanupLearner(ctx, m.Slot.Allocation.JobID, func(ctx context.Context, q nodeagent.CleanupRequest) error {
				calls++
				if q.WorkloadInitialized != (stage == "request_lost_ack") {
					t.Error("cleanup invented initialized credential evidence")
				}
				d, ok := ctx.Deadline()
				if !ok || time.Until(d) <= 5*time.Second || time.Until(d) > nodeagent.LearnerCleanupTimeout {
					t.Error("cleanup retained the hidden five-second cap")
				}
				return DestroyNodeWorkload(ctx, f.root, f.authority.Anchor, q)
			}))
			available, err := s.Availability(testNow)
			check(t, err)
			if !available.Available || calls != 1 {
				t.Fatal("partial credential preparation stranded allocation")
			}
		})
	}
}
