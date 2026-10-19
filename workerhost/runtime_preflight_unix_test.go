//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestLocalRunnerPreflightFailureFreezesRuntimeBeforePersistence(t *testing.T) {
	_, approval, guardian := learnerGuardianFixture(t)
	runner := guardian.local
	// Fail before the supervisor installs its runtime context or worker log.
	runner.supervisor.build = strings.Repeat("0", 64)
	ctx, cancel := attemptRuntimeContext(context.Background(), time.Hour)
	defer cancel()
	err := runner.Run(ctx, approval, func(int) error {
		t.Error("invalid build reached process start")
		return nil
	})
	if err == nil || errors.Is(err, errAttemptRuntime) {
		t.Fatalf("invalid preflight result: %v", err)
	}
	if _, err := os.Stat(filepath.Join(runner.directory.Dir(), "worker.log")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("fixture did not fail before runtime observer installation", err)
	}
	if _, err := os.Stat(filepath.Join(runner.directory.Dir(), physicalExitFile)); err != nil {
		t.Fatal("exit evidence was not persisted", err)
	}
	// Model a guardian observation after slow persistence crosses the deadline.
	// This must use the pre-persistence decision, not reclassify the failure.
	r := ctx.Value(attemptRuntimeKey{}).(*attemptRuntime)
	if cause := r.observe(ctx, r.deadline.Add(time.Hour)); cause != nil {
		t.Fatal("preflight failure was relabeled after persistence", cause)
	}
	if got := withRuntimeDiagnostic(ctx, err); got != err {
		t.Fatal("guardian changed the frozen preflight failure", got)
	}
}
