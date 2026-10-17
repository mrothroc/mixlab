//go:build darwin || linux

package workerhost

import (
	"context"
	"sync"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
)

func TestLearnerStoppedReplyPreservesHostDispatcher(t *testing.T) {
	s, p := learnerPlan(t, "stopped-reply")
	p.Learner.RuntimeSeconds = 40
	p.StartupTimeout = 15 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 55*time.Second)
	defer cancel()
	var once sync.Once
	owned := make(chan *ownedProcess, 1)
	s.sample = func(_ context.Context, process *ownedProcess, _ string) (resourceUsage, error) {
		once.Do(func() { owned <- process })
		return resourceUsage{}, nil
	}
	resumed := make(chan error, 1)
	started, calls := 0, 0
	resumeStarted := false
	p.Started = func(int) error { started++; return nil }
	p.Relay = func(call context.Context, e wc.Envelope) (wc.Envelope, error) {
		calls++
		var process *ownedProcess
		select {
		case process = <-owned:
		case <-call.Done():
			return wc.Envelope{}, call.Err()
		}
		if err := process.suspend(call); err != nil {
			return wc.Envelope{}, err
		}
		resumeStarted = true
		go func() {
			timer := time.NewTimer(7 * time.Second)
			defer timer.Stop()
			select {
			case <-ctx.Done():
				resumed <- ctx.Err()
			case <-timer.C:
				// The ownership/reap fence prevents signaling a reused PID even
				// when the mutation under test prematurely terminates the child.
				resumed <- process.resume(ctx)
			}
		}()
		e.Kind = wc.KindRoundResponse
		return e, nil
	}
	r, err := s.Run(ctx, p)
	cancel()
	if resumeStarted {
		if resumeErr := <-resumed; err == nil && resumeErr != nil {
			t.Fatalf("resume failed: %v", resumeErr)
		}
	}
	if err != nil || !r.Ready || r.PID == 0 || calls != 1 || started != 1 {
		t.Fatalf("stopped reply lost original learner: result=%+v err=%v starts=%d calls=%d", r, err, started, calls)
	}
}
