package integration

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"
	"time"
)

// A delayed authenticated reply must be allowed to complete, not repeatedly
// canceled at five seconds. Otherwise retries alone cannot tolerate a loaded
// coordinator whose every request takes longer than that old deadline.
func TestLearnerSemanticLoadedCoordinatorReply(t *testing.T) {
	x := newLearnerRelay(t)
	h := retryRelayHandler(t, x)
	var calls atomic.Int32
	server := serveCapabilities(t, x.g.Policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		timer := time.NewTimer(32 * time.Second)
		defer timer.Stop()
		select {
		case <-r.Context().Done():
			return
		case <-timer.C:
		}
		h.ServeHTTP(w, r)
	}))
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	out, err := retryRelayClient(t, x, server.URL, nil).Poll(ctx)
	check(t, err)
	if calls.Load() != 1 || out.Assignment == nil || *out.Assignment != x.s.Assignments[0] {
		t.Fatalf("latency changed work or retried completed reply: calls=%d out=%+v", calls.Load(), out)
	}
	state, err := x.model.Repository.Load(ctx, x.s.Plan.Cohort.RunID)
	check(t, err)
	if state.Recovery != nil || len(state.Attempts) != 0 {
		t.Fatal("latency consumed a replacement attempt")
	}
}
