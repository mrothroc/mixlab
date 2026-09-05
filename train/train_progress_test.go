package train

import (
	"strings"
	"testing"
	"time"
)

func TestTrainingProgressExcludesCompleteStartup(t *testing.T) {
	for _, firstStep := range []int{0, 1000} {
		start := time.Unix(100, 0)
		var p trainingProgress
		// Initial compilation/training takes 47.8 seconds. Its subsequent
		// validation, diagnostics, and logging take another 6.8 seconds.
		elapsed, steps, rate := p.collected(start.Add(47800*time.Millisecond), 64000)
		if elapsed != 0 || steps != 0 || rate != 0 {
			t.Fatalf("startup has a steady estimate: %s, %d, %g", elapsed, steps, rate)
		}
		if got := formatProgressTiming(54600*time.Millisecond, elapsed, steps, firstStep, 127260); strings.Contains(got, "remaining") {
			t.Fatalf("startup ETA should be unavailable: %s", got)
		}
		p.afterStep(start.Add(54600 * time.Millisecond))
		for completed := 1; completed <= 64; completed++ {
			now := start.Add(54600*time.Millisecond + time.Duration(completed)*400*time.Millisecond)
			elapsed, steps, rate = p.collected(now, 64000)
			if elapsed != time.Duration(completed)*400*time.Millisecond || steps != completed || rate != 160000 {
				t.Fatalf("step %d: elapsed=%s steps=%d rate=%g", firstStep+completed, elapsed, steps, rate)
			}
			p.afterStep(now)
		}
	}
}

func TestTrainingProgressCountsVariableBatchesAndPeriodicOverhead(t *testing.T) {
	var p trainingProgress
	start := time.Unix(100, 0)
	p.afterStep(start)
	p.collected(start.Add(time.Second), 100)
	// Periodic validation remains part of wall throughput/ETA, unlike startup.
	p.afterStep(start.Add(3 * time.Second))
	elapsed, steps, rate := p.collected(start.Add(4*time.Second), 300)
	if elapsed != 4*time.Second || steps != 2 || rate != 100 {
		t.Fatalf("elapsed=%s steps=%d rate=%g", elapsed, steps, rate)
	}
}

func TestTrainingProgressRequiresPositiveElapsed(t *testing.T) {
	var p trainingProgress
	start := time.Unix(100, 0)
	p.afterStep(start)
	elapsed, _, rate := p.collected(start, 100)
	if elapsed != 0 || rate != 0 {
		t.Fatalf("elapsed=%s rate=%g", elapsed, rate)
	}
}

func TestTrainingProgressStartupTelemetry(t *testing.T) {
	var p trainingProgress
	elapsed, _, rate := p.collected(time.Now(), 64000)
	state := newTelemetryState()
	state.update(telemetryUpdate{Step: 0, TotalSteps: 100, SteadyElapsed: elapsed, TokensPerSec: rate})
	snap := state.snapshot(false)
	if snap.TokensPerSec != 0 || snap.SteadyElapsedSeconds != 0 {
		t.Fatalf("startup telemetry has an estimate: %+v", snap)
	}
	if line := formatTelemetryLine(snap); !strings.Contains(line, "tok/s=n/a") {
		t.Fatalf("startup telemetry: %s", line)
	}
	if got := formatTrainingThroughput(160000); got != "160000" {
		t.Fatalf("steady throughput=%s", got)
	}
}
