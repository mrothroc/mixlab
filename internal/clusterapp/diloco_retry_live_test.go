package clusterapp

import (
	"context"
	"errors"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/nodeagent"
)

func committedRetryService(t *testing.T) (DiLoCoService, *admissionLeaseFixture) {
	t.Helper()
	s, saga, f := admissionMaintenanceService(t)
	s.Interval = time.Second
	for range 30 {
		v, err := saga.Snapshot(context.Background())
		check(t, err)
		if v.Done && v.Committed {
			return s, f
		}
		check(t, s.step(context.Background(), f.q.Launch.Run))
	}
	t.Fatal("fixture did not finish committed handoff")
	return s, f
}

func TestDiLoCoLiveRunErrorsKeepTickCadence(t *testing.T) {
	for _, persistent := range []bool{false, true} {
		t.Run(map[bool]string{false: "transient", true: "persistent"}[persistent], func(t *testing.T) {
			s, f := committedRetryService(t)
			ctx := context.Background()
			calls, events, round := 0, 0, 0
			repo := &recoveryTestRepository{state: f.state}
			models := &DiLoCoModels{Repository: repo}
			s.RunTerminal = models.Terminal
			s.Run = func(context.Context, admission.Submission, admission.Snapshot) error {
				calls++
				// Progress can coexist with a persistent joined recovery error.
				round++
				repo.state.Round = uint64(round)
				if persistent || calls == 1 {
					return syscall.EHOSTUNREACH
				}
				return nil
			}
			s.Event = func(string, error) { events++ }
			start := f.now.Load()
			for second := int64(0); second < 180; second++ {
				f.now.Store(start + second)
				check(t, s.step(ctx, f.q.Launch.Run))
				if calls != int(second+1) || round != calls {
					t.Fatalf("live tick delayed at second %d: calls=%d round=%d", second, calls, round)
				}
				o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
				check(t, err)
				if o.RetryAt != 0 || o.RetryFailures != 0 || o.RetryPhase != "" {
					t.Fatalf("live run scheduled backoff: %+v", o)
				}
			}
			wantEvents := 1
			if persistent {
				wantEvents = 3
			}
			if events != wantEvents {
				t.Fatalf("events=%d want %d", events, wantEvents)
			}
		})
	}
}

func TestDiLoCoLiveRunIgnoresPersistedCommittedBackoff(t *testing.T) {
	s, f := committedRetryService(t)
	ctx := context.Background()
	models := &DiLoCoModels{Repository: &recoveryTestRepository{state: f.state}}
	s.RunTerminal = models.Terminal
	at := f.now.Load()
	check(t, s.Inbox.ObserveState(ctx, f.q.Launch.Run, admission.Observation{
		At: at, Error: "previous live failure", EventAt: at,
		RetryAt: at + 60, RetryFailures: 16, RetryPhase: "committed",
	}))
	calls := 0
	s.Run = func(context.Context, admission.Submission, admission.Snapshot) error {
		calls++
		return nil
	}
	f.now.Store(at + 1)
	check(t, s.step(ctx, f.q.Launch.Run))
	o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
	check(t, err)
	if calls != 1 || o.RetryAt != 0 || o.Error != "" {
		t.Fatalf("stale live delay survived restart: calls=%d observation=%+v", calls, o)
	}
}

type retryReleaseNodes struct {
	*terminalLeaseNodes
	unreachable string
}

func (n *retryReleaseNodes) Status(ctx context.Context, a run.Allocation) (nodeagent.Lease, error) {
	if a.NodeID == n.unreachable {
		return nodeagent.Lease{}, &nodeFailure{node: a.NodeID, err: syscall.EHOSTUNREACH}
	}
	return n.terminalLeaseNodes.Status(ctx, a)
}

func TestDiLoCoRetryScopeUnavailableKeepsNormalTicks(t *testing.T) {
	for _, unavailable := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing_callback", true: "read_error"}[unavailable], func(t *testing.T) {
			s, f := committedRetryService(t)
			ctx := context.Background()
			if unavailable {
				s.RunTerminal = func(context.Context, string) (bool, error) {
					// Even a true value must not be trusted alongside an error.
					return true, errors.New("journal temporarily unavailable")
				}
			}
			calls := 0
			s.Run = func(context.Context, admission.Submission, admission.Snapshot) error {
				calls++
				return syscall.EHOSTUNREACH
			}
			start := f.now.Load()
			for second := int64(0); second < 10; second++ {
				f.now.Store(start + second)
				check(t, s.step(ctx, f.q.Launch.Run))
			}
			o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
			check(t, err)
			if calls != 10 || o.RetryAt != 0 {
				t.Fatalf("unknown state throttled work: calls=%d observation=%+v", calls, o)
			}
			if unavailable && !strings.Contains(o.Error, "journal temporarily unavailable") {
				t.Fatal("scope read failure hidden")
			}
		})
	}
}

func TestDiLoCoTerminalReleaseRetryAndRecovery(t *testing.T) {
	for _, phase := range []coordinator.Phase{coordinator.Complete, coordinator.Failed} {
		t.Run(string(phase), func(t *testing.T) {
			s, f := committedRetryService(t)
			ctx := context.Background()
			repo := &recoveryTestRepository{state: f.state}
			repo.state.Phase = phase
			if phase == coordinator.Complete {
				repo.state.Round = 1
				for i := range repo.state.PrivateHashes {
					repo.state.PrivateHashes[i] = f.plan.Contract.ConfigHash
				}
			} else {
				repo.state.Failure = "terminal fixture"
			}
			check(t, repo.state.Validate())
			nodes := &retryReleaseNodes{terminalLeaseNodes: lifecycleLeases(f), unreachable: f.plan.Slots[0].Allocation.NodeID}
			models := lifecycleModels(f, repo, nodes)
			s.RunTerminal = models.Terminal
			calls, events := 0, 0
			s.Run = func(ctx context.Context, q admission.Submission, v admission.Snapshot) error {
				calls++
				return models.Tick(ctx, q, v)
			}
			s.Event = func(string, error) { events++ }
			start := f.now.Load()
			end := start + 3600
			for due := start; due < end; {
				f.now.Store(due)
				before := calls
				restarted := s
				check(t, restarted.step(ctx, f.q.Launch.Run))
				if calls != before+1 {
					t.Fatal("due terminal retry was skipped")
				}
				o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
				check(t, err)
				base := min(int64(2)<<min(calls-1, 5), 48)
				if delay := o.RetryAt - due; delay < base || delay > base+base/4 {
					t.Fatalf("terminal delay did not grow/cap: attempt=%d delay=%d", calls, delay)
				}
				// Exercise the gate just after failure and just before the next
				// retry, then jump the fake clock. Polling every skipped second
				// only repeats durable-file reads, especially costly under race.
				lastProbe := due
				for _, probe := range []int64{due + 1, min(o.RetryAt-1, end-1)} {
					if probe <= lastProbe || probe >= end {
						continue
					}
					f.now.Store(probe)
					beforeEvents := events
					restarted = s
					check(t, restarted.step(ctx, f.q.Launch.Run))
					got, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
					check(t, err)
					if calls != before+1 || events != beforeEvents || got != o {
						t.Fatal("early probe bypassed persisted backoff")
					}
					lastProbe = probe
				}
				due = o.RetryAt
			}
			if calls < 10 || calls > 100 || events > 61 {
				t.Fatalf("terminal retry calls=%d events=%d", calls, events)
			}
			t.Logf("terminal %s, simulated outage hour: %d attempts, %d events", phase, calls, events)
			o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
			check(t, err)
			status := DiLoCoSubmissionStatus{Admission: "COMMITTED", RunPhase: phase}
			status.setObservation(o)
			if !status.AwaitingNode || status.NextRetryAt == 0 || status.ServiceNode != f.plan.Slots[0].Allocation.NodeID {
				t.Fatalf("terminal cleanup not visible: %+v", status)
			}
			if _, ok := models.released.Load(f.q.Launch.Run); ok {
				t.Fatal("unreachable node reported released")
			}
			if nodes.releases == 0 {
				t.Fatal("unreachable node blocked reachable-node release")
			}
			nodes.unreachable = ""
			for id, lease := range nodes.leases {
				lease.State = nodeagent.Released
				nodes.leases[id] = lease
			}
			f.now.Store(max(end, o.RetryAt))
			check(t, s.step(ctx, f.q.Launch.Run))
			if _, ok := models.released.Load(f.q.Launch.Run); !ok {
				t.Fatal("returning nodes did not confirm cleanup")
			}
			o, err = s.Inbox.Observation(ctx, f.q.Launch.Run)
			check(t, err)
			status.setObservation(o)
			if status.AwaitingNode || status.ServiceError || status.NextRetryAt != 0 {
				t.Fatalf("cleanup diagnostic not cleared: %+v", status)
			}
		})
	}
}
