package workerhost

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

var errAttemptRuntime = errors.New("worker attempt runtime limit reached")

type attemptRuntimeKey struct{}

// The guardian and supervisor share the outer runtime boundary and the first
// completion observation. Cleanup time must not change the recorded outcome.
type attemptRuntime struct {
	parent   context.Context
	deadline time.Time
	cause    error
	once     sync.Once
	expiry   error
}

// This is a per-attempt hosting clock, not a fresh allowance for each RPC.
// The caller's earlier cancellation/deadline still wins, including on recovery.
func attemptRuntimeContext(parent context.Context, runtime time.Duration) (context.Context, context.CancelFunc) {
	deadline := time.Now().Add(runtime)
	r := &attemptRuntime{parent: parent, deadline: deadline,
		cause: fmt.Errorf("%w (limit=%s deadline=%s)", errAttemptRuntime, runtime, deadline.UTC().Format(time.RFC3339Nano))}
	if outer, ok := parent.Value(attemptRuntimeKey{}).(*attemptRuntime); ok && !deadline.Before(outer.deadline) {
		if effective, bounded := parent.Deadline(); bounded && effective.Equal(outer.deadline) {
			r = outer
		}
	}
	ctx, cancel := context.WithDeadlineCause(parent, r.deadline, r.cause)
	return context.WithValue(ctx, attemptRuntimeKey{}, r), cancel
}

func (r *attemptRuntime) observe(ctx context.Context, now time.Time) error {
	r.once.Do(func() {
		cause := context.Cause(ctx)
		if errors.Is(cause, errAttemptRuntime) {
			r.expiry = cause
			return
		}
		if cause != nil || now.Before(r.deadline) {
			return
		}
		// An earlier parent limit or cancellation remains authoritative even if
		// neither context timer has been scheduled by the time socket I/O ends.
		if deadline, ok := r.parent.Deadline(); ok && deadline.Before(r.deadline) {
			return
		}
		if context.Cause(r.parent) != nil {
			return
		}
		r.expiry = r.cause
	})
	return r.expiry
}

func runtimeDiagnostic(ctx context.Context) error {
	if r, ok := ctx.Value(attemptRuntimeKey{}).(*attemptRuntime); ok {
		return r.observe(ctx, time.Now())
	}
	return nil
}

func withRuntimeDiagnostic(ctx context.Context, err error) error {
	cause := runtimeDiagnostic(ctx)
	if err == nil || errors.Is(err, errAttemptRuntime) || cause == nil {
		return err
	}
	// Keep protocol/resource/cleanup failures; expiry is additional local evidence,
	// not permission to reinterpret an EOF as a successful or retryable operation.
	return errors.Join(err, cause)
}
