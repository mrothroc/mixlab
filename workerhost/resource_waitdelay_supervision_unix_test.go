//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestLearnerResourceWaitDelaySuspendsResumesSameProcess(t *testing.T) {
	s, plan := learnerPlan(t, "wait-delay")
	plan.Learner.RuntimeSeconds = 30
	plan.StartupTimeout = 10 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	var starts atomic.Int32
	plan.Started = func(int) error { starts.Add(1); return nil }
	releaseRelay, releaseHealthy := make(chan struct{}), make(chan struct{})
	var relayOnce, healthyOnce sync.Once
	plan.Relay = func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
		select {
		case <-releaseRelay:
			e.Kind = wc.KindRoundResponse
			return e, nil
		case <-ctx.Done():
			return wc.Envelope{}, ctx.Err()
		}
	}
	postStop := make(chan *ownedProcess, 1)
	delays := make(chan delayedDrainResult, 2)
	var calls atomic.Int32
	s.sample = func(probe context.Context, process *ownedProcess, dir string) (resourceUsage, error) {
		switch calls.Add(1) {
		case 1, 2:
			r, fixtureErr := delayedDrainSample(probe, 0, false)
			delays <- r
			if fixtureErr != nil {
				return resourceUsage{}, errors.Join(r.err, fixtureErr)
			}
			return resourceUsage{}, r.err
		case 3:
			postStop <- process
			select {
			case <-releaseHealthy:
			case <-ctx.Done():
				return resourceUsage{}, ctx.Err()
			}
		}
		return sampleResources(probe, process, dir)
	}
	type outcome struct {
		result Result
		err    error
	}
	done := make(chan outcome, 1)
	go func() { r, err := s.Run(ctx, plan); done <- outcome{r, err} }()
	joined := false
	defer func() {
		cancel()
		healthyOnce.Do(func() { close(releaseHealthy) })
		relayOnce.Do(func() { close(releaseRelay) })
		if !joined {
			select {
			case <-done:
			case <-time.After(10 * time.Second):
				t.Error("learner supervision did not join after cancellation")
			}
		}
	}()
	var process *ownedProcess
	select {
	case process = <-postStop:
	case out := <-done:
		joined = true
		t.Fatalf("drain delay terminated learner before suspension: %+v", out)
	case <-ctx.Done():
		t.Fatal("learner never reached post-STOP sampling", ctx.Err())
	}
	for range 2 {
		r := <-delays
		if r.state == nil || !r.state.Success() || !errors.Is(r.err, exec.ErrWaitDelay) || errors.Is(r.err, context.DeadlineExceeded) || r.output != "" {
			t.Fatalf("supervision did not receive a real normal-exit drain delay: %+v", r)
		}
	}
	pid := process.cmd.Process.Pid
	awaitGroupStopped(t, []int{pid}, true)
	if ctx.Err() != nil || starts.Load() != 1 {
		t.Fatalf("slow observations canceled/restarted learner: ctx=%v starts=%d", ctx.Err(), starts.Load())
	}
	healthyOnce.Do(func() { close(releaseHealthy) })
	awaitGroupStopped(t, []int{pid}, false)
	select {
	case out := <-done:
		joined = true
		t.Fatalf("learner ended before completing its original relay: %+v", out)
	default:
	}
	relayOnce.Do(func() { close(releaseRelay) })
	select {
	case out := <-done:
		joined = true
		if out.err != nil || !out.result.Ready || out.result.PID != pid || starts.Load() != 1 || out.result.Exit.Kind != contract.ProcessExited || out.result.Exit.Code != 0 {
			t.Fatalf("drain delay lost successful same-process completion: result=%+v err=%v starts=%d", out.result, out.err, starts.Load())
		}
	case <-ctx.Done():
		t.Fatal("resumed learner did not complete", ctx.Err())
	}
}

func TestLearnerResourceWaitDelayDoesNotHideFailures(t *testing.T) {
	for _, tc := range []struct {
		name  string
		err   error
		usage resourceUsage
	}{
		{name: "unproven-drain-error", err: exec.ErrWaitDelay},
		{name: "joined-permission", err: errors.Join(&processSampleTimeoutError{cause: exec.ErrWaitDelay}, os.ErrPermission)},
		{name: "known-memory-breach", err: &processSampleTimeoutError{cause: exec.ErrWaitDelay}, usage: resourceUsage{Processes: []processUsage{{Identity: "worker", RSS: 101}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				control := &resourceControlFixture{}
				calls := 0
				sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
					calls++
					return tc.usage, tc.err
				}
				started := time.Now()
				stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{MemoryBytes: 100}, sample, control)
				<-ctx.Done()
				err := stop()
				if !errors.Is(err, tc.err) || calls != 1 || control.stops != 0 || control.resumes != 0 || time.Since(started) != 0 {
					t.Fatalf("unproven/mixed failure received grace: err=%v calls=%d control=%+v", err, calls, control)
				}
				if tc.usage.Processes != nil {
					var limit *ResourceLimitError
					if !errors.As(err, &limit) || limit.Resource != "memory_bytes" {
						t.Fatalf("known resource breach was hidden: %v", err)
					}
				}
			})
		})
	}
}
