//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func physicalFixture(t *testing.T) (*AttemptStore, contract.Approved, *LocalRunner) {
	t.Helper()
	s, a := attemptFixture(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	supervisor, err := New(exe, a.Assignment.BuildID)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewLocalRunner(supervisor, s.path, 3*time.Second, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	return s, a, r
}

func TestLocalRunnerDurableExecution(t *testing.T) {
	s, a, r := physicalFixture(t)
	ctx := context.Background()
	o, err := s.Run(ctx, a, r)
	if err != nil || o.Kind != contract.Exited || !o.NoChild || o.PID == 0 || syscall.Kill(o.PID, 0) != syscall.ESRCH {
		t.Fatalf("physical outcome: %+v %v", o, err)
	}
	status, err := s.Status()
	if err != nil || status.Started == nil || status.Started.PID != o.PID {
		t.Fatalf("missing durable start: %+v %v", status, err)
	}
	if again, err := s.Run(ctx, a, r); err != nil || again != o {
		t.Fatalf("journal retry launched/changed result: %+v %v", again, err)
	}
	if err := r.Run(ctx, a, func(int) error { t.Error("duplicate spawn"); return nil }); !errors.Is(err, ErrReconciliationRequired) {
		t.Fatalf("physical retry not fenced: %v", err)
	}
	// Model interruption after physical receipt but before terminal publication.
	status.Terminal = nil
	if gone, err := r.Reconcile(ctx, status); err != nil || !gone {
		t.Fatalf("lost physical exit evidence: %v %v", gone, err)
	}
	if err := os.Remove(filepath.Join(s.path.Dir(), physicalExitFile)); err != nil {
		t.Fatal(err)
	}
	if gone, err := r.Reconcile(ctx, status); err != nil || gone {
		t.Fatalf("invented cleanup from dead/reused PID: %v %v", gone, err)
	}
	boot, err := hostBootIdentity()
	if err != nil {
		t.Fatal(err)
	}
	other := "11111111-1111-1111-1111-111111111111"
	if boot == other {
		other = "22222222-2222-2222-2222-222222222222"
	}
	r.boot = func() (string, error) { return other, nil }
	if gone, err := r.Reconcile(ctx, status); err != nil || !gone {
		t.Fatalf("reboot did not fence old processes: %v %v", gone, err)
	}
	if err := os.Remove(filepath.Join(s.path.Dir(), physicalClaimFile)); err != nil {
		t.Fatal(err)
	}
	if gone, err := r.Reconcile(ctx, status); err == nil || gone {
		t.Fatalf("missing history accepted: %v %v", gone, err)
	}
}

func TestLocalRunnerStartPublicationBeforeAssignment(t *testing.T) {
	_, a, r := physicalFixture(t)
	a.Assignment.DatasetSelector = "resource-final-disk"
	var pid int
	err := r.Run(context.Background(), a, func(id int) error {
		pid = id
		return errors.New("start journal unavailable")
	})
	if err == nil || !strings.Contains(err.Error(), "start journal unavailable") || pid == 0 || syscall.Kill(pid, 0) != syscall.ESRCH {
		t.Fatalf("start publication failure not reaped: pid=%d %v", pid, err)
	}
	if _, err := os.Stat(filepath.Join(r.directory.Dir(), "large-output.bin")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("worker got assignment before start publication: %v", err)
	}
}

func TestLocalRunnerBudgetFailurePublishesFailed(t *testing.T) {
	s, a, r := physicalFixture(t)
	a.Assignment.DatasetSelector = "runtime-stall"
	r.supervisor.sample = func(context.Context, *ownedProcess, string) (resourceUsage, error) {
		return resourceUsage{Processes: []processUsage{{"worker", 0, a.Limits.MemoryBytes + 1}}}, nil
	}
	o, err := s.Run(context.Background(), a, r)
	if err != nil || o.Kind != contract.Failed || !o.NoChild || !strings.Contains(o.Error, "memory_bytes limit exceeded") {
		t.Fatalf("resource failure not preserved: %+v %v", o, err)
	}
}
