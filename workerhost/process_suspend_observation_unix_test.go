//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

func TestGroupStoppedObservationRetries(t *testing.T) {
	for _, slow := range []bool{false, true} {
		t.Run(map[bool]string{false: "failed-sample", true: "slow-sample"}[slow], func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 3*groupStateSampleTimeout)
				defer cancel()
				calls := 0
				err := waitGroupStopped(ctx, 2, true, func(probe context.Context) ([]byte, error) {
					calls++
					if calls == 1 {
						if slow {
							<-probe.Done()
							return nil, probe.Err()
						}
						return nil, errors.New("ps failed")
					}
					if calls == 2 {
						return []byte("11 S\n12 T\n"), nil
					}
					return []byte("11 T\n12 T\n"), nil
				})
				if err != nil || calls != 3 {
					t.Fatalf("observation did not recover: calls=%d err=%v", calls, err)
				}
			})
		})
	}
}

func TestGroupStoppedObservationBoundedFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 3*groupStateSampleTimeout)
		defer cancel()
		calls := 0
		err := waitGroupStopped(ctx, 1, false, func(probe context.Context) ([]byte, error) {
			calls++
			<-probe.Done()
			return nil, errors.New("ps killed")
		})
		if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "ps killed") || calls != 3 || time.Since(start) != 3*groupStateSampleTimeout {
			t.Fatalf("unbounded or misleading failure: calls=%d elapsed=%v err=%v", calls, time.Since(start), err)
		}
	})
}
