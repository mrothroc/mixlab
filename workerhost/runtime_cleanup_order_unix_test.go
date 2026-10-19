//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestSupervisorRuntimeObservationBeforeCleanupCancel(t *testing.T) {
	for _, expired := range []bool{false, true} {
		name := "before-boundary"
		if expired {
			name = "expired-timer-not-delivered"
		}
		t.Run(name, func(t *testing.T) {
			s, p := learnerPlan(t, "success")
			p.Learner.RuntimeSeconds = 3600
			ctx, cancel := attemptRuntimeContext(context.Background(), time.Hour)
			defer cancel()
			r := ctx.Value(attemptRuntimeKey{}).(*attemptRuntime)
			var timeout error
			p.Started = func(int) error {
				reader, writer := net.Pipe()
				defer func() { _ = reader.Close() }()
				defer func() { _ = writer.Close() }()
				if err := reader.SetReadDeadline(time.Now().Add(-time.Second)); err != nil {
					return err
				}
				_, timeout = reader.Read(make([]byte, 1))
				// The nested supervisor has already inherited this record. Move
				// only its observed boundary: both real context timers remain an
				// hour away. The callback and deferred observation are sequential.
				if expired {
					r.deadline = time.Now().Add(-time.Second)
				}
				return timeout
			}
			result, err := s.Run(ctx, p)
			if timeout == nil || !errors.Is(timeout, os.ErrDeadlineExceeded) || !errors.Is(err, timeout) {
				t.Fatalf("fixture did not preserve the I/O timeout: %v / %v", timeout, err)
			}
			if context.Cause(ctx) != nil {
				t.Fatal("outer context timer fired; test no longer pins delayed delivery")
			}
			if errors.Is(err, errAttemptRuntime) != expired {
				t.Fatalf("cleanup cancellation changed runtime classification: %v", err)
			}
			if errors.Is(runtimeDiagnostic(ctx), errAttemptRuntime) != expired {
				t.Fatal("outer observation disagrees with supervisor cleanup")
			}
			if result.PID == 0 || syscall.Kill(result.PID, 0) != syscall.ESRCH {
				t.Fatal("supervisor did not reap its child")
			}
			log, readErr := os.ReadFile(filepath.Join(p.Directory.Dir(), "worker.log"))
			if readErr != nil || strings.Contains(string(log), "worker attempt runtime limit reached") != expired {
				t.Fatalf("worker log disagrees with frozen observation: %q / %v", log, readErr)
			}
		})
	}
}
