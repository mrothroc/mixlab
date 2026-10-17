package admission

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
)

func TestAdmissionLoadedEffectRetainsDeadlineBounds(t *testing.T) {
	for _, mode := range []string{"slow-success", "caller-deadline", "admission-deadline"} {
		t.Run(mode, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				x := setup(t)
				ctx := context.Background()
				if _, err := x.s.Step(ctx, x.p); err != nil {
					t.Fatal(err)
				}
				v, err := x.s.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if mode == "admission-deadline" {
					x.now = v.Deadline - 10
				}
				base, start := x.now, time.Now()
				x.p.Clock = func() int64 { return base + int64(time.Since(start)/time.Second) }
				if mode == "caller-deadline" {
					var cancel context.CancelFunc
					ctx, cancel = context.WithTimeout(ctx, 10*time.Second)
					defer cancel()
				}
				reserve := x.p.Reserve
				calls := 0
				x.p.Reserve = func(call context.Context, c recruitment.Candidate, q nodeagent.Reserve) (nodeagent.Lease, error) {
					calls++
					want := StepTimeout
					if mode != "slow-success" {
						want = 10 * time.Second
					}
					deadline, ok := call.Deadline()
					if !ok || time.Until(deadline) != want {
						t.Fatalf("effect deadline = %v, want %s", deadline, want)
					}
					select {
					case <-call.Done():
						return nodeagent.Lease{}, call.Err()
					case <-time.After(40 * time.Second):
						x.now = x.p.Clock()
						return reserve(call, c, q)
					}
				}
				_, err = x.s.Step(ctx, x.p)
				if calls != 1 {
					t.Fatal("effect was not attempted exactly once", calls)
				}
				if mode == "slow-success" {
					if err != nil || time.Since(start) != 40*time.Second {
						t.Fatalf("loaded effect truncated: elapsed=%s err=%v", time.Since(start), err)
					}
				} else if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) != 10*time.Second {
					t.Fatalf("effect escaped authority: elapsed=%s err=%v", time.Since(start), err)
				}
			})
		})
	}
}
