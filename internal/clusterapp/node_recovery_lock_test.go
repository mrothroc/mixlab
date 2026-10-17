package clusterapp

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func TestNodeRecoveryLockDeadline(t *testing.T) {
	for _, operation := range []string{"authorize", "start"} {
		t.Run(operation, func(t *testing.T) {
			// Nil dependencies deliberately ensure no work happens after timeout.
			a := &NodeJobs{}
			a.mu.Lock()
			defer a.mu.Unlock()
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
			defer cancel()
			done := make(chan error, 1)
			go func() {
				if operation == "authorize" {
					_, err := a.AuthorizeLearnerRetry(ctx, trust.AuthenticatedPrincipal{}, "", NodeLearnerRetryRequest{})
					done <- err
				} else {
					done <- a.StartLearnerRetry(ctx, trust.AuthenticatedPrincipal{}, "", NodeLearnerRetryStart{})
				}
			}()
			select {
			case err := <-done:
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("expected request deadline, got %v", err)
				}
				w := httptest.NewRecorder()
				nodeResponse(&nodeResponseContext{ResponseWriter: w, ctx: ctx}, nil, err)
				if w.Code != http.StatusServiceUnavailable {
					t.Fatalf("deadline response=%d", w.Code)
				}
			case <-time.After(time.Second):
				t.Fatal("recovery waited for the unrelated operation after deadline")
			}
		})
	}
}

func TestNodeRecoveryLockAcquiresAndHonorsCancellation(t *testing.T) {
	a := &NodeJobs{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := a.lockRecovery(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if !a.mu.TryLock() {
		t.Fatal("canceled request acquired the mutex")
	}
	done := make(chan error, 1)
	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	wait := &recoveryLockWaitContext{Context: ctx, waiting: make(chan struct{})}
	go func() { done <- a.lockRecovery(wait) }()
	select {
	case <-wait.waiting:
	case <-ctx.Done():
		a.mu.Unlock()
		<-done
		t.Fatal("request did not reach contended lock wait")
	}
	a.mu.Unlock()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if a.mu.TryLock() {
		t.Fatal("successful acquisition did not retain the serialization fence")
	}
	a.mu.Unlock()
}

type recoveryLockWaitContext struct {
	context.Context
	waiting chan struct{}
	once    sync.Once
}

func (c *recoveryLockWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.waiting) })
	return c.Context.Done()
}
