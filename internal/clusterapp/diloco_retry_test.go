package clusterapp

import (
	"context"
	"fmt"
	"syscall"
	"testing"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
)

func TestDiLoCoUnavailableNodeRetryIsBoundedAndRecovers(t *testing.T) {
	s, saga, f := admissionMaintenanceService(t)
	ctx := context.Background()
	id := f.q.Launch.Run
	var attempts []int64
	events := 0
	up := false
	build := s.Admission
	s.Admission = func(ctx context.Context, q admission.Submission) (admission.Ports, error) {
		p, err := build(ctx, q)
		p.Reserve = func(context.Context, recruitment.Candidate, nodeagent.Reserve) (nodeagent.Lease, error) {
			attempts = append(attempts, f.now.Load())
			return nodeagent.Lease{}, syscall.EHOSTUNREACH
		}
		p.AbortReservation = func(context.Context, recruitment.Candidate, nodeagent.Reserve) error {
			attempts = append(attempts, f.now.Load())
			if !up {
				return syscall.EHOSTUNREACH
			}
			return nil
		}
		return p, err
	}
	s.Event = func(string, error) { events++ }
	start := f.now.Load()
	for second := int64(0); second < 3600; second++ {
		f.now.Store(start + second)
		if second == 30 {
			check(t, saga.Cancel(ctx))
		}
		// Reconstruct the service each turn as a stronger restart regression:
		// retry pacing cannot depend on actor-local counters.
		restarted := s
		check(t, restarted.step(ctx, id))
	}
	if len(attempts) > 100 || events > 61 {
		t.Fatalf("unbounded outage: attempts=%d events=%d", len(attempts), events)
	}
	t.Logf("one simulated hour: %d node attempts, %d service error events", len(attempts), events)
	if len(attempts) < 10 {
		t.Fatal("retry stopped permanently")
	}
	for i := 1; i < min(4, len(attempts)); i++ {
		if attempts[i]-attempts[i-1] < int64(1<<i) {
			t.Fatalf("backoff did not grow: %v", attempts[:4])
		}
	}
	v, err := saga.Snapshot(ctx)
	check(t, err)
	if v.Done || !v.Aborting {
		t.Fatal("unreachable reservation was reported aborted")
	}
	up = true
	for second := int64(3600); second < 3730; second++ {
		f.now.Store(start + second)
		check(t, s.step(ctx, id))
	}
	v, err = saga.Snapshot(ctx)
	check(t, err)
	if !v.Done || v.Committed {
		t.Fatal("returning node did not complete abort")
	}
}

func TestDiLoCoLeaseOutageIsBackedOff(t *testing.T) {
	s, _, f := admissionMaintenanceService(t)
	calls := 0
	s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
		calls++
		return syscall.ECONNREFUSED
	}
	start := f.now.Load()
	for second := int64(0); second < 100; second++ {
		f.now.Store(start + second)
		check(t, s.step(context.Background(), f.q.Launch.Run))
	}
	if calls > 10 || calls < 4 {
		t.Fatalf("maintenance retries=%d", calls)
	}
}

func TestDiLoCoRetryJitterAndChangingErrorsStayBounded(t *testing.T) {
	seen := map[int64]bool{}
	for i := range 100 {
		for count := uint32(1); count <= 16; count++ {
			delay := admissionRetryDelay(fmt.Sprint(i), count, 100)
			if delay < 2 || delay > 60 {
				t.Fatal("unbounded delay", delay)
			}
			if count == 16 {
				seen[delay] = true
			}
		}
	}
	if len(seen) < 2 {
		t.Fatal("no per-run jitter")
	}
	previous := admission.Observation{}
	events := 0
	for second := int64(1000); second < 4600; second++ {
		next, emit := serviceObservation("run", second, previous, admission.Snapshot{}, admission.Snapshot{}, true, fmt.Errorf("changing failure %d", second))
		if emit {
			events++
		}
		previous = next
	}
	if events != 60 {
		t.Fatal("changing text bypassed log limit", events)
	}
}

func TestDiLoCoExpiredAdmissionBypassesRetryOnce(t *testing.T) {
	s, saga, f := admissionMaintenanceService(t)
	ctx := context.Background()
	f.now.Store(f.q.Deadline - 1)
	s.Maintain = func(context.Context, admission.Submission, admission.Snapshot) error {
		if f.now.Load() >= f.q.Deadline {
			return ErrAdmissionLeaseLost
		}
		return syscall.EHOSTUNREACH
	}
	check(t, s.step(ctx, f.q.Launch.Run))
	o, err := s.Inbox.Observation(ctx, f.q.Launch.Run)
	check(t, err)
	if o.RetryAt <= f.q.Deadline {
		t.Fatal("fixture did not cross deadline")
	}
	f.now.Store(f.q.Deadline)
	check(t, s.step(ctx, f.q.Launch.Run))
	v, err := saga.Snapshot(ctx)
	check(t, err)
	if !v.Aborting {
		t.Fatal("retry wait hid admission expiry")
	}
}
