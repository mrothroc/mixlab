package workerhost

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func TestAttemptRuntimeDiagnosticRetainsCause(t *testing.T) {
	ctx, cancel := attemptRuntimeContext(context.Background(), time.Millisecond)
	defer cancel()
	<-ctx.Done()
	for _, original := range []error{io.EOF, context.DeadlineExceeded, errors.New("resource limit exceeded")} {
		err := withRuntimeDiagnostic(ctx, original)
		if !errors.Is(err, original) || !errors.Is(err, errAttemptRuntime) || !strings.Contains(err.Error(), "limit=1ms deadline=") {
			t.Fatalf("runtime diagnostic lost cause or bound: %v", err)
		}
		if again := withRuntimeDiagnostic(ctx, err); again != err {
			t.Fatal("nested supervisor duplicated runtime diagnostic")
		}
	}
	if withRuntimeDiagnostic(ctx, nil) != nil {
		t.Fatal("diagnostic changed a successful result")
	}
}

func TestAttemptRuntimeDiagnosticDoesNotInventExpiry(t *testing.T) {
	for _, mode := range []string{"live", "parent-canceled", "parent-deadline", "local-canceled"} {
		t.Run(mode, func(t *testing.T) {
			parent, stop := context.WithCancel(context.Background())
			defer stop()
			if mode == "parent-deadline" {
				var cancel context.CancelFunc
				parent, cancel = context.WithDeadline(parent, time.Now().Add(-time.Second))
				defer cancel()
			}
			ctx, cancel := attemptRuntimeContext(parent, time.Hour)
			defer cancel()
			switch mode {
			case "parent-canceled":
				stop()
			case "local-canceled":
				cancel()
			}
			if err := withRuntimeDiagnostic(ctx, io.EOF); err != io.EOF {
				t.Fatalf("unrelated EOF relabeled as runtime expiry: %v", err)
			}
			if mode == "parent-deadline" && ctx.Err() != context.DeadlineExceeded {
				t.Fatal("attempt reset earlier parent deadline")
			}
		})
	}
}
