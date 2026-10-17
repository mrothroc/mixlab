//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func sampleTimeout(ctx context.Context) (resourceUsage, error) {
	<-ctx.Done()
	return resourceUsage{}, &processSampleTimeoutError{cause: ctx.Err()}
}

func TestResourceTimeoutHealthyRecoveryPreservesCPU(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			calls++
			switch calls {
			case 1:
				return resourceUsage{Processes: []processUsage{
					{Identity: "worker", CPU: 2 * time.Second},
					{Identity: "exited-child", CPU: time.Second},
				}}, nil
			case 2:
				return sampleTimeout(ctx)
			default:
				return resourceUsage{Processes: []processUsage{{Identity: "worker", CPU: time.Duration(calls) * time.Second}}}, nil
			}
		}
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{CPUSeconds: 4, MemoryBytes: 100, DiskBytes: 100}, sample)
		time.Sleep(3500 * time.Millisecond)
		synctest.Wait()
		if ctx.Err() != nil || calls != 3 {
			_ = stop()
			t.Fatalf("isolated timeout did not recover: calls=%d err=%v", calls, ctx.Err())
		}
		// The healthy sample after the gap observes 3s + the exited child's 1s.
		// Dropping the CPU baseline double-counts it; resetting consumed CPU
		// misses the following breach at 5s.
		time.Sleep(time.Second)
		synctest.Wait()
		if ctx.Err() == nil {
			_ = stop()
			t.Fatal("CPU accumulated across the gap was not enforced")
		}
		var limit *ResourceLimitError
		if err := stop(); !errors.As(err, &limit) || limit.Resource != "cpu_nanoseconds" || limit.Observed != int64(5*time.Second) {
			t.Fatalf("CPU accounting after gap: %v", err)
		}
	})
}

func TestResourceConsecutiveTimeoutsAbort(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			calls++
			return sampleTimeout(ctx)
		}
		start := time.Now()
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
		<-ctx.Done()
		if err := stop(); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("second timeout did not fail closed: %v", err)
		}
		if calls != 2 || time.Since(start) != 2*resourceSampleTimeout {
			t.Fatalf("unexpected timeout tolerance: calls=%d elapsed=%s", calls, time.Since(start))
		}
	})
}

func TestResourceGraceBoundsStalledSampler(t *testing.T) {
	for _, initialHealthy := range []bool{false, true} {
		name := "startup"
		if initialHealthy {
			name = "after-healthy"
		}
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				release := make(chan struct{})
				calls := 0
				sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
					calls++
					if !initialHealthy || calls != 1 {
						// Model uninterruptible filesystem I/O. Supervision must
						// cancel before joining this sampler.
						<-release
					}
					return resourceUsage{}, nil
				}
				start := time.Now()
				stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
				<-ctx.Done()
				elapsed := time.Since(start)
				close(release)
				if err := stop(); !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "no healthy resource sample") {
					t.Fatalf("freshness failure suppressed by cancellation: %v", err)
				}
				if elapsed != resourceSampleGrace {
					t.Fatalf("unbounded/stretched freshness gap: %s", elapsed)
				}
			})
		})
	}
}

func TestResourceHealthySampleResetsTolerance(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			calls++
			if calls%2 == 1 {
				return sampleTimeout(ctx)
			}
			return resourceUsage{}, nil
		}
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
		time.Sleep(10500 * time.Millisecond)
		synctest.Wait()
		if ctx.Err() != nil || calls < 6 {
			_ = stop()
			t.Fatalf("healthy samples did not reset tolerance/freshness: calls=%d err=%v", calls, ctx.Err())
		}
		if err := stop(); err != nil {
			t.Fatalf("shutdown during tolerated gap: %v", err)
		}
	})
}

func TestResourcePermanentErrorsNeverReceiveGrace(t *testing.T) {
	malformed := errors.New("malformed process resource sample")
	for _, failure := range []error{
		malformed,
		os.ErrPermission,
		context.DeadlineExceeded, // Disk timeout or unproven sampler deadline.
		errors.Join(&processSampleTimeoutError{cause: context.DeadlineExceeded}, os.ErrPermission),
	} {
		t.Run(failure.Error(), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				start := time.Now()
				calls := 0
				sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
					calls++
					return resourceUsage{}, failure
				}
				stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
				<-ctx.Done()
				if err := stop(); !errors.Is(err, failure) || calls != 1 || time.Since(start) != 0 {
					t.Fatalf("permanent failure received grace: calls=%d elapsed=%s err=%v", calls, time.Since(start), err)
				}
			})
		})
	}
}

func TestResourceShutdownRetainsPermanentSampleFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		sample := func(context.Context, *ownedProcess, string) (resourceUsage, error) {
			cancel()
			return resourceUsage{}, errors.Join(os.ErrPermission, context.Canceled)
		}
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
		<-ctx.Done()
		if err := stop(); !errors.Is(err, os.ErrPermission) {
			t.Fatalf("shutdown suppressed permanent sampler failure: %v", err)
		}
	})
}

func TestResourceCallerDeadlineIsNotFreshnessFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			<-ctx.Done()
			return resourceUsage{}, ctx.Err()
		}
		start := time.Now()
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{}, sample)
		<-ctx.Done()
		if err := stop(); err != nil || time.Since(start) != time.Second {
			t.Fatalf("caller deadline became freshness failure: elapsed=%s err=%v", time.Since(start), err)
		}
	})
}

func TestResourceKnownBreachAfterTimeoutStillImmediate(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		calls := 0
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			calls++
			if calls == 1 {
				return sampleTimeout(ctx)
			}
			cancel() // A known breach also wins over concurrent shutdown.
			return resourceUsage{Processes: []processUsage{{Identity: "worker", RSS: 101}}}, nil
		}
		stop := watchResources(ctx, cancel, nil, "", contract.Limits{MemoryBytes: 100}, sample)
		<-ctx.Done()
		var limit *ResourceLimitError
		if err := stop(); !errors.As(err, &limit) || limit.Resource != "memory_bytes" || calls != 2 {
			t.Fatalf("known breach received grace: calls=%d err=%v", calls, err)
		}
	})
}
