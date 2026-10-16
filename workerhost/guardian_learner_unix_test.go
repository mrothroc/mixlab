//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func learnerGuardianFixture(t *testing.T) (*AttemptStore, contract.Approved, *GuardianRunner) {
	t.Helper()
	s, p := learnerPlan(t, "success")
	l, err := NewLocalRunner(s, p.Directory, 5*time.Second, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	l.LearnerRelay, l.LearnerPrivateDirectory = p.Relay, p.LearnerPrivateDirectory
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewGuardianRunner(l, exe, s.build)
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewAttemptStore(p.Directory)
	if err != nil {
		t.Fatal(err)
	}
	a := contract.Approved{Version: contract.LearnerVersion, ManifestHash: strings.Repeat("a", 64), TransportHash: p.Learner.PlanHash, Learner: p.Learner, Limits: p.Limits}
	return store, a, r
}

func TestGuardianLearnerOpaqueRelayAndDurableOutcome(t *testing.T) {
	t.Setenv("WORKERHOST_SECRET", "credential-canary")
	s, a, r := learnerGuardianFixture(t)
	var calls atomic.Int32
	r.local.LearnerRelay = func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
		calls.Add(1)
		if e.PayloadKind != "owner_opaque_v99" {
			return wc.Envelope{}, errors.New("payload interpreted by host")
		}
		select {
		case <-time.After(100 * time.Millisecond):
		case <-ctx.Done():
			return wc.Envelope{}, ctx.Err()
		}
		e.Kind = wc.KindRoundResponse
		return e, nil
	}
	out, err := s.Run(context.Background(), a, r)
	if err != nil || out.Kind != contract.Exited || !out.NoChild || out.PID == 0 || calls.Load() != 1 {
		t.Fatalf("learner guardian: %+v calls=%d err=%v", out, calls.Load(), err)
	}
	if syscall.Kill(out.PID, 0) != syscall.ESRCH {
		t.Fatal("learner not reaped")
	}
	again, err := s.Run(context.Background(), a, r)
	if err != nil || again != out || calls.Load() != 1 {
		t.Fatalf("retry relaunched learner: %+v %v", again, err)
	}
}

func TestGuardianLearnerCancelDuringRelay(t *testing.T) {
	s, a, r := learnerGuardianFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	r.local.LearnerRelay = func(call context.Context, _ wc.Envelope) (wc.Envelope, error) {
		cancel()
		<-call.Done()
		return wc.Envelope{}, call.Err()
	}
	out, err := s.Run(ctx, a, r)
	if err != nil || !out.NoChild || out.Kind != contract.Canceled || out.PID == 0 {
		t.Fatalf("cancel lost cleanup: %+v %v", out, err)
	}
	if syscall.Kill(out.PID, 0) != syscall.ESRCH {
		t.Fatal("canceled learner not reaped")
	}
}

func TestLearnerApprovalRejectsMixedSchemas(t *testing.T) {
	_, a, _ := learnerGuardianFixture(t)
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	_, ddp, _ := guardianFixture(t)
	a.Assignment = ddp.Assignment
	if a.Validate() == nil {
		t.Fatal("mixed DDP/learner approval accepted")
	}
	a.Assignment = contract.Approved{}.Assignment
	a.Version = contract.Version
	if a.Validate() == nil {
		t.Fatal("learner accepted as DDP approval version")
	}
}
