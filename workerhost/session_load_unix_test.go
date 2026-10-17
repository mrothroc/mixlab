//go:build darwin || linux

package workerhost

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

// Exercise the real spawned process, authenticated local socket, dispatcher and
// supervision lanes. This delay exceeds the former 20s HTTP / 30s IPC budgets;
// it must retain the same PID/attempt while heartbeats continue independently.
func TestLearnerSessionSlowRelayRetainsProcess(t *testing.T) {
	s, p := learnerPlan(t, "slow-relay")
	p.Learner.RuntimeSeconds = 90
	p.StartupTimeout = 10 * time.Second
	started := 0
	p.Started = func(int) error { started++; return nil }
	calls := 0
	p.Relay = func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
		calls++
		timer := time.NewTimer(32 * time.Second)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return wc.Envelope{}, ctx.Err()
		case <-timer.C:
		}
		e.Kind = wc.KindRoundResponse
		return e, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Second)
	defer cancel()
	r, err := s.Run(ctx, p)
	if err != nil || !r.Ready || started != 1 || calls != 1 {
		t.Fatalf("slow relay lost process: result=%+v err=%v starts=%d calls=%d", r, err, started, calls)
	}
}

func TestLearnerResourcePauseBeforeReadinessRetainsProcess(t *testing.T) {
	s, p := learnerPlan(t, "startup-load")
	p.Learner.RuntimeSeconds = 90
	p.StartupTimeout = 2 * time.Minute
	ctx, cancel := context.WithTimeout(context.Background(), 80*time.Second)
	defer cancel()
	var calls atomic.Int32
	s.sample = func(context.Context, *ownedProcess, string) (resourceUsage, error) {
		if calls.Add(1) == 1 {
			// A blocked sample forces STOP at the freshness limit. The next
			// complete sample must CONT the same child, still awaiting admission.
			select {
			case <-time.After(32 * time.Second):
			case <-ctx.Done():
				return resourceUsage{}, ctx.Err()
			}
		}
		return resourceUsage{}, nil
	}
	starts := 0
	p.Started = func(int) error {
		starts++
		// Hold admission beyond the old 30s bootstrap deadline and until
		// the stale pre-STOP observation has been replaced by a healthy one.
		select {
		case <-time.After(35 * time.Second):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	r, err := s.Run(ctx, p)
	if err != nil || !r.Ready || starts != 1 || calls.Load() < 2 {
		t.Fatalf("pre-readiness suspension lost attempt: result=%+v err=%v starts=%d samples=%d", r, err, starts, calls.Load())
	}
}
