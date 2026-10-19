package workerhost

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

// Socket deadlines can expire before the context's timer callback runs. This
// context deliberately exposes the deadline without firing a cancellation.
type delayedDeadlineContext struct {
	context.Context
	deadline time.Time
}

func (c delayedDeadlineContext) Deadline() (time.Time, bool) { return c.deadline, true }

func TestAttemptRuntimeObservationBeforeTimerDelivery(t *testing.T) {
	deadline := time.Unix(1000, 0)
	for _, tc := range []struct {
		name   string
		now    time.Time
		expiry bool
	}{
		{"before", deadline.Add(-time.Nanosecond), false},
		{"exact", deadline, true},
		{"after", deadline.Add(time.Second), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &attemptRuntime{parent: context.Background(), deadline: deadline, cause: errAttemptRuntime}
			ctx := context.WithValue(delayedDeadlineContext{context.Background(), deadline}, attemptRuntimeKey{}, r)
			if context.Cause(ctx) != nil || ctx.Err() != nil {
				t.Fatal("fixture delivered timer before observation")
			}
			if got := r.observe(ctx, tc.now); errors.Is(got, errAttemptRuntime) != tc.expiry {
				t.Fatalf("absolute boundary lost: %v", got)
			}
			err := withRuntimeDiagnostic(ctx, io.EOF)
			if !errors.Is(err, io.EOF) || errors.Is(err, errAttemptRuntime) != tc.expiry {
				t.Fatalf("outcome disagrees with frozen boundary: %v", err)
			}
			if errors.Is(runtimeDiagnostic(ctx), errAttemptRuntime) != tc.expiry {
				t.Fatal("log classification disagrees with returned error")
			}
		})
	}
}

func TestAttemptRuntimeObservationDoesNotRelabelEarlierStops(t *testing.T) {
	deadline := time.Unix(1000, 0)
	for _, mode := range []string{"parent-deadline", "parent-canceled", "local-canceled"} {
		t.Run(mode, func(t *testing.T) {
			parent := context.Background()
			switch mode {
			case "parent-deadline":
				parent = delayedDeadlineContext{parent, deadline.Add(-time.Second)}
			case "parent-canceled":
				p, cancel := context.WithCancel(parent)
				cancel()
				parent = p
			}
			r := &attemptRuntime{parent: parent, deadline: deadline, cause: errAttemptRuntime}
			ctx, cancel := context.WithCancel(context.WithValue(parent, attemptRuntimeKey{}, r))
			defer cancel()
			if mode == "local-canceled" {
				cancel()
			}
			if got := r.observe(ctx, deadline.Add(time.Hour)); got != nil {
				t.Fatalf("earlier stop relabeled as attempt expiry: %v", got)
			}
		})
	}
}

func TestAttemptRuntimeObservationSurvivesCleanup(t *testing.T) {
	for _, expired := range []bool{false, true} {
		name := "early-failure"
		if expired {
			name = "expired"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := attemptRuntimeContext(context.Background(), time.Hour)
			defer cancel()
			r := ctx.Value(attemptRuntimeKey{}).(*attemptRuntime)
			observed := r.deadline.Add(-time.Second)
			if expired {
				observed = r.deadline
			}
			if cause := r.observe(ctx, observed); errors.Is(cause, errAttemptRuntime) != expired {
				t.Fatal("initial observation disagrees with boundary", cause)
			}
			cancel()
			// Cleanup cancellation wins the real timer race, but cannot replace
			// the already captured absolute-deadline observation.
			if context.Cause(ctx) != context.Canceled {
				t.Fatal("fixture did not capture cleanup cancellation")
			}
			if got := r.observe(ctx, r.deadline.Add(time.Hour)); errors.Is(got, errAttemptRuntime) != expired {
				t.Fatalf("cleanup changed observation: %v", got)
			}
			if got := withRuntimeDiagnostic(ctx, io.ErrUnexpectedEOF); !errors.Is(got, io.ErrUnexpectedEOF) || errors.Is(got, errAttemptRuntime) != expired {
				t.Fatalf("cleanup lost outcome: %v", got)
			}
		})
	}
}

func TestNestedAttemptRuntimePreservesCompletionObservation(t *testing.T) {
	outer, cancelOuter := attemptRuntimeContext(context.Background(), time.Hour)
	defer cancelOuter()
	inner, cancelInner := attemptRuntimeContext(outer, time.Hour)
	defer cancelInner()
	r := outer.Value(attemptRuntimeKey{}).(*attemptRuntime)
	if inner.Value(attemptRuntimeKey{}) != r {
		t.Fatal("supervisor reset guardian runtime boundary")
	}
	if got, _ := inner.Deadline(); !got.Equal(r.deadline) {
		t.Fatal("nested context extended runtime")
	}
	// Successful supervision followed by slow cleanup/persistence is not a new
	// runtime failure. The guardian must retain the earlier observation too.
	if err := withRuntimeDiagnostic(inner, nil); err != nil {
		t.Fatal("successful result changed")
	}
	cancelInner()
	if got := r.observe(outer, r.deadline.Add(time.Second)); got != nil {
		t.Fatal("guardian relabeled pre-deadline completion after cleanup")
	}
	if err := withRuntimeDiagnostic(outer, io.EOF); err != io.EOF {
		t.Fatal("guardian changed later cleanup error")
	}
}

func TestNestedAttemptRuntimeRespectsShorterBounds(t *testing.T) {
	outer, cancel := attemptRuntimeContext(context.Background(), time.Hour)
	defer cancel()
	for _, mode := range []string{"shorter-attempt", "shorter-parent"} {
		t.Run(mode, func(t *testing.T) {
			parent, duration := outer, time.Minute
			if mode == "shorter-parent" {
				var stop context.CancelFunc
				parent, stop = context.WithTimeout(outer, time.Minute)
				defer stop()
				duration = time.Hour
			}
			inner, stop := attemptRuntimeContext(parent, duration)
			defer stop()
			if inner.Value(attemptRuntimeKey{}) == outer.Value(attemptRuntimeKey{}) {
				t.Fatal("independent shorter bound reused outer observation")
			}
			if d, _ := inner.Deadline(); !d.Before(outer.Value(attemptRuntimeKey{}).(*attemptRuntime).deadline) {
				t.Fatal("shorter bound was lost")
			}
		})
	}
}
