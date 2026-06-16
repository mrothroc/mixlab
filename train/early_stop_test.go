package train

import "testing"

func TestEarlyStopPatience(t *testing.T) {
	state := newEarlyStopState(&EarlyStopSpec{Patience: 2, MinDelta: 0.01})
	if stop, reason := state.observe(0, 1.00); stop || reason != "" {
		t.Fatalf("first observation stopped: stop=%v reason=%q", stop, reason)
	}
	if stop, reason := state.observe(1, 0.995); stop || reason != "" {
		t.Fatalf("small non-improvement 1 stopped: stop=%v reason=%q", stop, reason)
	}
	stop, reason := state.observe(2, 0.994)
	if !stop || reason == "" {
		t.Fatalf("second stale observation stop=%v reason=%q, want stop", stop, reason)
	}
}

func TestEarlyStopStepGatedValGT(t *testing.T) {
	state := newEarlyStopState(&EarlyStopSpec{ValGT: 1.5, AtStep: 3})
	if stop, reason := state.observe(2, 1.8); stop || reason != "" {
		t.Fatalf("pre-gate observation stopped: stop=%v reason=%q", stop, reason)
	}
	stop, reason := state.observe(3, 1.8)
	if !stop || reason == "" {
		t.Fatalf("step-gated val_gt stop=%v reason=%q, want stop", stop, reason)
	}
}

func TestEarlyStopMinStepsDelaysPatienceCount(t *testing.T) {
	state := newEarlyStopState(&EarlyStopSpec{Patience: 1, MinSteps: 3})
	if stop, reason := state.observe(0, 1.0); stop || reason != "" {
		t.Fatalf("first observation stopped: stop=%v reason=%q", stop, reason)
	}
	if stop, reason := state.observe(1, 1.1); stop || reason != "" {
		t.Fatalf("pre-min-steps stale observation stopped: stop=%v reason=%q", stop, reason)
	}
	stop, reason := state.observe(3, 1.1)
	if !stop || reason == "" {
		t.Fatalf("post-min-steps stale observation stop=%v reason=%q, want stop", stop, reason)
	}
}
