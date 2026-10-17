//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

type resourceControlFixture struct {
	stops, resumes int
	paused         bool
	stopErr        error
	resumeErr      error
}

func (c *resourceControlFixture) suspend(context.Context) error {
	c.stops++
	c.paused = true
	return c.stopErr
}

func (c *resourceControlFixture) resume(ctx context.Context) error {
	if ctx.Err() == nil {
		c.resumes++
		c.paused = false
	}
	return c.resumeErr
}

func TestLearnerResourceLongStallSuspendsWithoutRestart(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		control := &resourceControlFixture{}
		releaseOld, releaseHealthy := make(chan struct{}), make(chan struct{})
		calls := 0
		sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
			calls++
			switch calls {
			case 1:
				<-releaseOld // A filesystem sample blocked well beyond 6s.
			case 2:
				<-releaseHealthy
			}
			return resourceUsage{}, nil
		}
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{}, sample, control)
		time.Sleep(30 * time.Second)
		synctest.Wait()
		if ctx.Err() != nil || control.stops != 1 || !control.paused || calls != 1 {
			t.Errorf("stall killed worker or lost bounded supervision: ctx=%v control=%+v calls=%d", ctx.Err(), control, calls)
		}
		close(releaseOld)
		synctest.Wait()
		if control.resumes != 0 || !control.paused || calls != 2 {
			t.Errorf("pre-suspension sample resumed worker: control=%+v calls=%d", control, calls)
		}
		close(releaseHealthy)
		synctest.Wait()
		if ctx.Err() != nil || control.resumes != 1 || control.paused {
			t.Errorf("healthy post-suspension sample did not resume: ctx=%v control=%+v", ctx.Err(), control)
		}
		if err := stop(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestLearnerResourceRepeatedTimeoutsPreserveCPU(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		control := &resourceControlFixture{}
		var calls atomic.Int64
		var healthy atomic.Bool
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			if calls.Add(1) == 1 {
				return resourceUsage{Processes: []processUsage{
					{Identity: "worker", CPU: 2 * time.Second}, {Identity: "child", CPU: time.Second},
				}}, nil
			}
			if !healthy.Load() {
				return sampleTimeout(ctx)
			}
			cpu := 3 * time.Second
			if control.resumes > 0 {
				cpu = 4 * time.Second
			}
			return resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: cpu}}}, nil
		}
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{CPUSeconds: 4}, sample, control)
		time.Sleep(30 * time.Second)
		synctest.Wait()
		if ctx.Err() != nil || control.stops != 1 || control.resumes != 0 || calls.Load() < 10 {
			t.Errorf("timeouts became loss or stopped retrying: ctx=%v control=%+v calls=%d", ctx.Err(), control, calls.Load())
		}
		healthy.Store(true)
		time.Sleep(1500 * time.Millisecond)
		synctest.Wait()
		if ctx.Err() != nil || control.resumes != 1 {
			t.Errorf("CPU baseline lost across suspension: ctx=%v control=%+v", ctx.Err(), control)
		}
		time.Sleep(time.Second)
		synctest.Wait()
		var limit *ResourceLimitError
		if err := stop(); !errors.As(err, &limit) || limit.Resource != "cpu_nanoseconds" || limit.Observed != int64(5*time.Second) {
			t.Fatalf("cumulative CPU breach lost after resume: %v", err)
		}
	})
}

func TestLearnerResourceSampleStraddlingStopCannotResume(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		control := &resourceControlFixture{}
		postStop := make(chan struct{})
		calls := 0
		sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
			calls++
			switch calls {
			case 2:
				// Delay delivering the first timeout until t=5s. The next
				// probe crosses the freshness boundary but meets its own deadline.
				time.Sleep(4 * time.Second)
				return resourceUsage{}, &processSampleTimeoutError{cause: context.DeadlineExceeded}
			case 3:
				time.Sleep(1500 * time.Millisecond)
			case 4:
				<-postStop
			}
			return resourceUsage{}, nil
		}
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{}, sample, control)
		time.Sleep(6750 * time.Millisecond)
		synctest.Wait()
		if ctx.Err() != nil || calls != 4 || control.stops != 1 || control.resumes != 0 || !control.paused {
			t.Errorf("straddling sample resumed worker: calls=%d control=%+v ctx=%v", calls, control, ctx.Err())
		}
		close(postStop)
		synctest.Wait()
		if control.resumes != 1 || control.paused {
			t.Errorf("post-STOP sample did not resume: %+v", control)
		}
		if err := stop(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestLearnerResourceDiskDeadlinesSuspend(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		control := &resourceControlFixture{}
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			<-ctx.Done()
			return resourceUsage{}, ctx.Err()
		}
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{}, sample, control)
		time.Sleep(20 * time.Second)
		synctest.Wait()
		if ctx.Err() != nil || !control.paused || control.stops != 1 {
			t.Errorf("disk sampling delay killed learner: ctx=%v control=%+v", ctx.Err(), control)
		}
		if err := stop(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestLearnerResourceSuspensionDoesNotExtendLifetime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
		defer cancel()
		control := &resourceControlFixture{}
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			return sampleTimeout(ctx)
		}
		start := time.Now()
		stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{}, sample, control)
		<-ctx.Done()
		if err := stop(); err != nil || time.Since(start) != 40*time.Second || !control.paused || control.resumes != 0 {
			t.Fatalf("suspension changed lifetime: elapsed=%s control=%+v err=%v", time.Since(start), control, err)
		}
	})
}

func TestLearnerResourceFatalEvidenceWhileSuspended(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage resourceUsage
		err   error
	}{
		{name: "memory", usage: resourceUsage{Processes: []processUsage{{Identity: "worker", RSS: 101}}}},
		{name: "malformed", err: errors.New("malformed process sample")},
		{name: "permission", err: os.ErrPermission},
		{name: "joined-timeout-permission", err: errors.Join(&processSampleTimeoutError{cause: context.DeadlineExceeded}, os.ErrPermission)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				control := &resourceControlFixture{}
				calls := 0
				sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
					calls++
					if calls <= 2 {
						return sampleTimeout(ctx)
					}
					return tc.usage, tc.err
				}
				stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{MemoryBytes: 100}, sample, control)
				<-ctx.Done()
				err := stop()
				if err == nil || calls != 3 || control.stops != 1 || control.resumes != 0 {
					t.Fatalf("fatal evidence resumed worker: calls=%d control=%+v err=%v", calls, control, err)
				}
				if tc.err != nil && !errors.Is(err, tc.err) {
					t.Fatalf("lost permanent evidence: %v", err)
				}
			})
		})
	}
}

func TestLearnerResourceSignalFailureIsFatal(t *testing.T) {
	for _, resumeFailure := range []bool{false, true} {
		name := "stop"
		if resumeFailure {
			name = "resume"
		}
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				control := &resourceControlFixture{stopErr: os.ErrPermission}
				if resumeFailure {
					control.stopErr, control.resumeErr = nil, os.ErrPermission
				}
				calls := 0
				sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
					calls++
					if calls <= 2 {
						return sampleTimeout(ctx)
					}
					return resourceUsage{}, nil
				}
				stop := watchResourceMonitor(ctx, cancel, nil, "", contract.Limits{}, sample, control)
				<-ctx.Done()
				if err := stop(); !errors.Is(err, os.ErrPermission) {
					t.Fatalf("signal failure ignored: %v", err)
				}
			})
		})
	}
}
