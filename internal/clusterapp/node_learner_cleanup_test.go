package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/workerhost"
)

func TestLearnerCleanupTimeoutClassification(t *testing.T) {
	terminal := errors.New("credential identity mismatch")
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"deadline", context.DeadlineExceeded, true},
		{"cancellation", context.Canceled, true},
		{"wrapped", fmt.Errorf("cleanup: %w", context.DeadlineExceeded), true},
		{"joined_timeouts", errors.Join(context.DeadlineExceeded, context.Canceled), true},
		{"terminal", terminal, false},
		{"mixed", errors.Join(context.DeadlineExceeded, terminal), false},
		{"nested_mixed", fmt.Errorf("cleanup: %w", errors.Join(context.Canceled, terminal)), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := contextOnlyFailure(tc.err); got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestUnpreparedCleanupJoinTimeoutIsRetryable(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		loop := nodeExecutionLoop{ports: NodeExecutionPorts{StopPreparation: func(ctx context.Context, job string) error {
			if job != "" {
				t.Fatal("unprepared cleanup must stop the active preparation")
			}
			<-ctx.Done()
			return ctx.Err()
		}}}
		err := loop.cleanupUnprepared(context.Background(), "unused")
		if !errors.Is(err, workerhost.ErrReconciliationRequired) || !errors.Is(err, context.DeadlineExceeded) || time.Since(start) != nodeagent.LearnerCleanupTimeout {
			t.Fatalf("unbounded or terminal cleanup: elapsed=%v err=%v", time.Since(start), err)
		}
	})
}
