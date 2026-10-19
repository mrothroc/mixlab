//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestGuardianRuntimeDiagnosticBeforeSupervisor(t *testing.T) {
	_, approval, runner := learnerGuardianFixture(t)
	approval.Learner.RuntimeSeconds = 1
	dir := runner.local.directory
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	var runErr error
	var startedPID int
	// Hold the real cross-process lock through the guardian's runtime and its
	// bounded reconciliation attempt. LocalRunner cannot reach Supervisor.Run,
	// so only the guardian can explain this deadline in its terminal message.
	err := dir.WithProcessLock(ctx, physicalOwnerLock, func() error {
		runErr = runner.Run(ctx, approval, func(pid int) error {
			startedPID = pid
			return errors.New("unexpected worker start behind physical owner lock")
		})
		return nil
	})
	if err != nil {
		t.Fatal("physical owner lock", err)
	}
	if ctx.Err() != nil {
		t.Fatal("guardian outlived bounded test context", runErr)
	}
	if runErr == nil || !strings.Contains(runErr.Error(), "worker attempt runtime limit reached") || !strings.Contains(runErr.Error(), "limit=1s deadline=") {
		t.Fatalf("pre-supervisor expiry lost guardian diagnostic: %v", runErr)
	}
	if startedPID != 0 {
		t.Fatalf("worker started despite held physical lock: %d", startedPID)
	}
	for _, name := range []string{physicalClaimFile, physicalExitFile, "worker.log", "control.sock"} {
		if _, err := os.Lstat(filepath.Join(dir.Dir(), name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("supervisor/physical ownership artifact %q exists: %v", name, err)
		}
	}
	// No local ownership receipt was inspected or created. Do not turn the
	// test's knowledge that no child started into fabricated cleanup authority.
	if !errors.Is(runErr, ErrReconciliationRequired) {
		t.Fatalf("pre-owner failure incorrectly claimed reconciled cleanup: %v", runErr)
	}
	_, guardian, err := loadGuardian(dir)
	if err != nil || guardian.State != "owned" {
		t.Fatalf("guardian did not retain unresolved ownership: %+v %v", guardian, err)
	}
	var exit *exec.ExitError
	if !errors.As(runErr, &exit) || exit.ProcessState == nil || !exit.Exited() || exit.ExitCode() != 1 {
		t.Fatalf("guardian exit was not joined: %v", runErr)
	}
	if err := syscall.Kill(exit.Pid(), 0); err != syscall.ESRCH {
		t.Fatalf("guardian still exists after joined exit: %v", err)
	}
	lockCtx, stopLocks := context.WithTimeout(context.Background(), 2*time.Second)
	defer stopLocks()
	for _, name := range []string{physicalOwnerLock, guardianLock} {
		if err := dir.WithProcessLock(lockCtx, name, func() error { return nil }); err != nil {
			t.Fatalf("lock %q leaked after guardian exit: %v", name, err)
		}
	}
}
