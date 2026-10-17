//go:build darwin || linux

package workerhost

import (
	"context"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

type freshnessControlFixture struct {
	stops, resumes atomic.Int32
	paused         atomic.Bool
}

func (c *freshnessControlFixture) suspend(context.Context) error {
	c.stops.Add(1)
	c.paused.Store(true)
	return nil
}

func (c *freshnessControlFixture) resume(context.Context) error {
	c.resumes.Add(1)
	c.paused.Store(false)
	return nil
}

func TestLearnerResourceResultCannotWinAgainstStaleClock(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		control := &freshnessControlFixture{}
		clockCalls := 0
		var samples atomic.Int32
		now := func() time.Time {
			clockCalls++
			if clockCalls == 1 {
				return time.Now()
			}
			// Model selection of an already-ready result after a scheduling
			// stall, before the freshness timer notification is serviced.
			return time.Now().Add(resourceSampleGrace + time.Second)
		}
		sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
			samples.Add(1)
			return resourceUsage{Processes: []processUsage{{Identity: "owned", RSS: 1}}}, ctx.Err()
		}
		done := make(chan error, 1)
		go func() {
			done <- monitorResources(ctx, cancel, nil, "", contract.Limits{MemoryBytes: 1}, sample, control, now)
		}()
		synctest.Wait()
		if samples.Load() != 1 || control.stops.Load() != 1 || control.resumes.Load() != 0 || !control.paused.Load() {
			t.Errorf("ready result bypassed stale observation: samples=%d stops=%d resumes=%d paused=%v", samples.Load(), control.stops.Load(), control.resumes.Load(), control.paused.Load())
		}
		time.Sleep(resourceInterval)
		synctest.Wait()
		if samples.Load() != 2 || control.stops.Load() != 1 || control.resumes.Load() != 1 || control.paused.Load() {
			t.Errorf("fresh post-STOP observation did not resume: samples=%d stops=%d resumes=%d paused=%v", samples.Load(), control.stops.Load(), control.resumes.Load(), control.paused.Load())
		}
		cancel()
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	})
}
