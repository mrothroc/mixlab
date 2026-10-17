//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func retainedDiskPlan() LaunchPlan {
	return LaunchPlan{
		Learner: &workerjob.LearnerAssignment{PrivateMaxBytes: 100},
		Limits:  contract.Limits{DiskBytes: 200},
	}
}

func TestLearnerFinalDiskAllowsSlowVerificationWithinRemainingLifetime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// Model the original supervision context, not a fresh post-exit budget.
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		deadline, _ := ctx.Deadline()
		time.Sleep(10 * time.Second)
		calls := 0
		sample := func(ctx context.Context, _ string) (int64, error) {
			calls++
			if got, ok := ctx.Deadline(); !ok || !got.Equal(deadline) {
				t.Errorf("verification replaced original deadline: got=%s want=%s", got, deadline)
			}
			time.Sleep(3 * time.Second) // Each directory exceeds the old 2s limit.
			return 50, ctx.Err()
		}
		start := time.Now()
		if err := retainedDiskPlan().checkLearnerDisk(ctx, sample); err != nil || calls != 2 || time.Since(start) != 6*time.Second {
			t.Fatalf("slow retained-output check failed: calls=%d elapsed=%s err=%v", calls, time.Since(start), err)
		}
	})
}

func TestLearnerFinalDiskDoesNotRestartRuntime(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		time.Sleep(18 * time.Second)
		start := time.Now()
		calls := 0
		sample := func(ctx context.Context, _ string) (int64, error) {
			calls++
			<-ctx.Done()
			return 0, ctx.Err()
		}
		if err := retainedDiskPlan().checkLearnerDisk(ctx, sample); !errors.Is(err, context.DeadlineExceeded) || calls != 1 || time.Since(start) != 2*time.Second {
			t.Fatalf("final check extended runtime: calls=%d elapsed=%s err=%v", calls, time.Since(start), err)
		}
	})
}

func TestLearnerFinalDiskHonorsParentCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		parent, revoke := context.WithCancel(context.Background())
		defer revoke()
		ctx, cancel := context.WithTimeout(parent, time.Minute)
		defer cancel()
		time.AfterFunc(3*time.Second, revoke)
		start := time.Now()
		sample := func(ctx context.Context, _ string) (int64, error) {
			<-ctx.Done()
			return 0, ctx.Err()
		}
		if err := retainedDiskPlan().checkLearnerDisk(ctx, sample); !errors.Is(err, context.Canceled) || time.Since(start) != 3*time.Second {
			t.Fatalf("verification outlived cancellation: elapsed=%s err=%v", time.Since(start), err)
		}
	})
}

func TestLearnerFinalDiskRetainsPartialLimitEvidence(t *testing.T) {
	for _, private := range []bool{false, true} {
		name := "attempt"
		if private {
			name = "private"
		}
		t.Run(name, func(t *testing.T) {
			p := retainedDiskPlan()
			calls := 0
			sample := func(context.Context, string) (int64, error) {
				calls++
				if private && calls == 1 {
					return 50, nil
				}
				return 201, context.DeadlineExceeded
			}
			var limit *ResourceLimitError
			if err := p.checkLearnerDisk(context.Background(), sample); !errors.As(err, &limit) || !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("partial disk breach lost: %v", err)
			}
		})
	}
}

func TestLearnerPrivateDiskPreservesPartialTotal(t *testing.T) {
	p := retainedDiskPlan()
	sample := func(context.Context, string) (int64, error) { return 75, context.DeadlineExceeded }
	n, err := p.withPrivateDiskSample(context.Background(), 50, sample)
	if n != 125 || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("partial disk total lost: n=%d err=%v", n, err)
	}
}
