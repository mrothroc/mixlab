package learneripc

import (
	"context"
	"errors"
	"io"
	"testing"
	"testing/synctest"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
)

func TestSemanticWaitRetainsStatePastAttemptBudgets(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
		defer cancel()
		start := time.Now()
		calls, controls := 0, 0
		done := make(chan struct{})
		go func() {
			defer close(done)
			ticker := time.NewTicker(time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					controls++
				}
			}
		}()
		out, err := waitSemantic(ctx, func() (response, error) {
			calls++
			if calls <= 4 {
				time.Sleep(20 * time.Second)
				return response{}, private.ErrUnavailable
			}
			return response{Reply: &private.RelayResponse{Done: true}}, nil
		})
		cancel()
		<-done
		if err != nil || out.Reply == nil || !out.Reply.Done || calls != 5 || time.Since(start) < 80*time.Second || controls < 75 {
			t.Fatalf("waiting lost state/control liveness: out=%+v err=%v calls=%d controls=%d", out, err, calls, controls)
		}
	})
}

func TestSemanticWaitCancellationAndPermanentErrors(t *testing.T) {
	for _, failure := range []error{io.ErrUnexpectedEOF, errors.New("authorization rejected")} {
		calls := 0
		_, err := waitSemantic(context.Background(), func() (response, error) { calls++; return response{}, failure })
		if err != failure || calls != 1 {
			t.Fatal("permanent/stream failure retried", calls, err)
		}
	}
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_, err := waitSemantic(ctx, func() (response, error) { return response{}, private.ErrUnavailable })
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal("job deadline did not stop waiting", err)
		}
	})
}
