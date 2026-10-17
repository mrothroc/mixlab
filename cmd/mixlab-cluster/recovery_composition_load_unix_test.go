//go:build darwin || linux

package main

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
)

// Production installed service/agent/guardian composition with real child
// processes and mTLS. Only numerical work is synthetic. Delayed authenticated
// node replies exceed the old 10s route / 15s client budgets. A 65s coordinator
// outage also exhausts the new network retry cycle; both original children must
// resume semantic polling without consuming a replacement attempt.
func TestInstalledLearnerLoadRetainsAttempts(t *testing.T) {
	x := newInstalledRecoveryFixtureMode(t, false, true)
	defer x.networkPause.release()
	deadline := time.NewTimer(150 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	type identity struct {
		attempt string
		pid     int
	}
	initial := map[string]identity{}
	var started time.Time
	released := false
	for {
		s, err := x.repo.Load(x.ctx, x.submission.Launch.Run)
		if err == nil && s.Phase == coordinator.Failed {
			t.Fatalf("load failed run: %s; %s", s.Failure, x.diagnostics())
		}
		if err == nil && s.Phase == coordinator.Open {
			compositionCheck(t, os.WriteFile(x.pollGate, []byte("open"), 0600))
			if len(s.Attempts) != 0 || s.Recovery != nil {
				t.Fatalf("latency caused replacement: %+v; %s", s.Attempts, x.diagnostics())
			}
			for nodeID, node := range x.nodes {
				e, err := node.ActiveExecution(x.ctx)
				compositionCheck(t, err)
				if e == nil || e.Outcome == nil || e.Outcome.PID == 0 || e.Learner == nil {
					if _, known := initial[nodeID]; known {
						t.Fatalf("original process disappeared: %s; %s", nodeID, x.diagnostics())
					}
					continue
				}
				current := identity{e.Learner.Assignment.AttemptID, e.Outcome.PID}
				if prior, ok := initial[nodeID]; ok && (prior != current || e.Learner.Stopped) {
					t.Fatalf("load lost original process: %s %+v -> %+v; %s", nodeID, prior, current, x.diagnostics())
				}
				initial[nodeID] = current
			}
			if len(initial) == 2 && len(s.Starts) == 2 && started.IsZero() {
				for _, proxy := range x.proxies {
					proxy.mu.Lock()
					proxy.latency = 20 * time.Second
					proxy.mu.Unlock()
				}
				started = time.Now()
				x.networkPause.start()
			}
			if !released && !started.IsZero() && time.Since(started) >= 65*time.Second {
				for _, proxy := range x.proxies {
					proxy.mu.Lock()
					n, proxyErr := proxy.delayed, proxy.err
					proxy.latency = 0
					proxy.mu.Unlock()
					if n < 2 || proxyErr != nil {
						t.Fatalf("latency not exercised cleanly: requests=%d error=%v", n, proxyErr)
					}
				}
				if x.networkPause.waits.Load() < 4 {
					t.Fatal("coordinator outage did not exhaust both learners' network attempts")
				}
				pattern := strings.TrimSuffix(x.pollGate, ".composition-open")
				compositionCheck(t, os.WriteFile(pattern+".composition-resume", []byte("resume"), 0600))
				x.networkPause.release()
				released = true
			}
			if released {
				resumed := 0
				pattern := strings.TrimSuffix(x.pollGate, ".composition-open")
				for _, slot := range s.Plan.Slots {
					b, err := os.ReadFile(pattern + ".composition-after-" + slot.Member.WorkerID)
					if err == nil && string(b) == initial[slot.Allocation.NodeID].attempt {
						resumed++
					}
				}
				if resumed == 2 {
					return
				}
			}
		}
		select {
		case <-x.ctx.Done():
			t.Fatalf("fixture expired: %s", x.diagnostics())
		case <-deadline.C:
			t.Fatalf("load acceptance timed out: %s", x.diagnostics())
		case <-ticker.C:
		}
	}
}
