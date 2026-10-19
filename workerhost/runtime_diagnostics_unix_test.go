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

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestLearnerAttemptRuntimeDiagnostic(t *testing.T) {
	s, p := learnerPlan(t, "success")
	p.Learner.RuntimeSeconds = 1
	p.StartupTimeout = 3 * time.Second
	p.Relay = func(ctx context.Context, _ wc.Envelope) (wc.Envelope, error) {
		<-ctx.Done()
		return wc.Envelope{}, ctx.Err()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := s.Run(ctx, p)
	if !errors.Is(err, errAttemptRuntime) || !strings.Contains(err.Error(), "limit=1s deadline=") {
		t.Fatalf("runtime expiry remained opaque: result=%+v err=%v", result, err)
	}
	if result.PID == 0 || syscall.Kill(result.PID, 0) != syscall.ESRCH {
		t.Fatal("runtime diagnostic changed child cleanup")
	}
	log, err := os.ReadFile(filepath.Join(p.Directory.Dir(), "worker.log"))
	if err != nil || !strings.Contains(string(log), "workerhost: worker attempt runtime limit reached") {
		t.Fatalf("local worker log lacks runtime explanation: %q %v", log, err)
	}
}

func TestGuardianLearnerRuntimeDiagnosticPersists(t *testing.T) {
	store, a, runner := learnerGuardianFixture(t)
	a.Learner.RuntimeSeconds = 1
	runner.local.LearnerRelay = func(ctx context.Context, _ wc.Envelope) (wc.Envelope, error) {
		<-ctx.Done()
		return wc.Envelope{}, ctx.Err()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := store.Run(ctx, a, runner)
	if err != nil || out.Kind != contract.Failed || !out.NoChild || !strings.Contains(out.Error, "worker attempt runtime limit reached") {
		t.Fatalf("guardian outcome lost runtime cause: %+v %v", out, err)
	}
	if out.PID == 0 || syscall.Kill(out.PID, 0) != syscall.ESRCH {
		t.Fatal("guardian runtime diagnostic lost physical cleanup")
	}
	again, err := store.Run(ctx, a, runner)
	if err != nil || again != out {
		t.Fatalf("retry changed terminal evidence: %+v %v", again, err)
	}
}
