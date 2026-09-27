//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func guardianFixture(t *testing.T) (*AttemptStore, contract.Approved, *GuardianRunner) {
	t.Helper()
	s, a, local := physicalFixture(t)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewGuardianRunner(local, exe, a.Assignment.BuildID)
	if err != nil {
		t.Fatal(err)
	}
	return s, a, r
}

func TestGuardianDurableExecution(t *testing.T) {
	s, a, r := guardianFixture(t)
	o, err := s.Run(context.Background(), a, r)
	if err != nil || o.Kind != contract.Exited || !o.NoChild || o.PID == 0 {
		t.Fatalf("guardian outcome: %+v %v", o, err)
	}
	if syscall.Kill(o.PID, 0) != syscall.ESRCH {
		t.Fatal("worker not reaped")
	}
	if again, err := s.Run(context.Background(), a, r); err != nil || again != o {
		t.Fatalf("retry changed outcome: %+v %v", again, err)
	}
}

func TestGuardianPendingClaimFenced(t *testing.T) {
	s, a, r := guardianFixture(t)
	_, hash, err := freezeApproval(a)
	if err != nil {
		t.Fatal(err)
	}
	boot, err := hostBootIdentity()
	if err != nil {
		t.Fatal(err)
	}
	claim := guardianClaim{hash, boot}
	b, err := json.Marshal(guardianRecord{claim, "pending"})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.path.WriteFile(guardianFile, b); err != nil {
		t.Fatal(err)
	}
	status := Attempt{Version: contract.Version, Approval: a, Digest: hash}
	if gone, err := r.Reconcile(context.Background(), status); err != nil || !gone {
		t.Fatalf("pending helper not fenced: %v %v", gone, err)
	}
	if err := transitionGuardian(context.Background(), s.path, claim, "pending", "owned"); err == nil {
		t.Fatal("delayed helper bypassed cancellation fence")
	}
}

func TestGuardianStartPublicationFailure(t *testing.T) {
	_, a, r := guardianFixture(t)
	a.Assignment.DatasetSelector = "resource-final-disk"
	var pid int
	err := r.Run(context.Background(), a, func(id int) error { pid = id; return errors.New("start publication fault") })
	if err == nil || errors.Is(err, ErrReconciliationRequired) || pid == 0 || syscall.Kill(pid, 0) != syscall.ESRCH {
		t.Fatalf("start fault cleanup: pid=%d %v", pid, err)
	}
	if _, err := os.Stat(filepath.Join(r.local.directory.Dir(), "large-output.bin")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("assignment escaped start fence: %v", err)
	}
}

func TestGuardianReconcilePhysicalReceiptCrashWindow(t *testing.T) {
	for _, test := range []struct {
		name       string
		receipt    bool
		wrongClaim bool
	}{
		{name: "missing_receipt_stays_fenced"},
		{name: "matching_receipt_recovers", receipt: true},
		{name: "foreign_receipt_rejected", receipt: true, wrongClaim: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			s, a, r := guardianFixture(t)
			_, hash, err := freezeApproval(a)
			if err != nil {
				t.Fatal(err)
			}
			boot, err := hostBootIdentity()
			if err != nil {
				t.Fatal(err)
			}
			claim := guardianClaim{hash, boot}
			b, err := json.Marshal(guardianRecord{claim, "owned"})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.path.WriteFile(guardianFile, b); err != nil {
				t.Fatal(err)
			}
			if test.receipt {
				p := physicalClaim{hash, boot}
				if test.wrongClaim {
					p.Approval = "foreign"
				}
				b, err := json.Marshal(physicalExit{p, 123})
				if err != nil {
					t.Fatal(err)
				}
				if err := s.path.WriteFile(physicalExitFile, b); err != nil {
					t.Fatal(err)
				}
			}
			gone, err := r.Reconcile(context.Background(), Attempt{Version: contract.Version, Approval: a, Digest: hash})
			if test.wrongClaim {
				if err == nil || gone {
					t.Fatalf("accepted foreign receipt: %v %v", gone, err)
				}
				return
			}
			if err != nil || gone != test.receipt {
				t.Fatalf("reconcile: %v %v", gone, err)
			}
			if test.receipt {
				if err := transitionGuardian(context.Background(), s.path, claim, "owned", "done"); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// helperGuardianAgent is a real sacrificial agent process. Killing it must not
// kill the helper first; the helper owns its child and the cleanup receipt.
func helperGuardianAgent(path string) error {
	dir, err := statehome.Resolve(statehome.Options{ExactDir: path}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return err
	}
	b, err := dir.ReadFileLimit("test-approval.json", 1<<20)
	if err != nil {
		return err
	}
	var a contract.Approved
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	s, err := New(exe, a.Assignment.BuildID)
	if err != nil {
		return err
	}
	l, err := NewLocalRunner(s, dir, 5*time.Second, 2*time.Second)
	if err != nil {
		return err
	}
	r, err := NewGuardianRunner(l, exe, a.Assignment.BuildID)
	if err != nil {
		return err
	}
	store, err := NewAttemptStore(dir)
	if err != nil {
		return err
	}
	_, err = store.Run(context.Background(), a, r)
	return err
}

func TestGuardianAgentDeathReconcilesLiveWorker(t *testing.T) {
	s, a, r := guardianFixture(t)
	a.Assignment.DatasetSelector = "runtime-stall"
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.path.WriteFile("test-approval.json", b); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(r.binary, "internal-test-agent", s.path.Dir())
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		if !waited {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	var status Attempt
	deadline := time.Now().Add(10 * time.Second)
	for {
		status, err = s.Status()
		if err == nil && status.Started != nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("agent never published worker start", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	pid := status.Started.PID
	for {
		if _, err := os.Stat(filepath.Join(s.path.Dir(), "worker-ready")); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("worker never entered uncooperative workload")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()
	waited = true
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for {
		out, err := s.Reconcile(ctx, r)
		if err == nil {
			if out.Kind != contract.Failed || !out.NoChild || syscall.Kill(pid, 0) != syscall.ESRCH {
				t.Fatalf("unsafe post-crash outcome: %+v", out)
			}
			if again, err := s.Run(ctx, a, r); err != nil || again != out {
				t.Fatalf("restart spawned duplicate: %+v %v", again, err)
			}
			break
		}
		if !errors.Is(err, ErrReconciliationRequired) || ctx.Err() != nil {
			t.Fatal(err)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestGuardianRequiresDescriptorAndExactSchema(t *testing.T) {
	cmd := exec.Command(os.Args[0], guardianCommand)
	if out, err := cmd.CombinedOutput(); err == nil {
		t.Fatalf("accepted missing descriptor: %s", out)
	}
	_, a, r := guardianFixture(t)
	_, hash, err := freezeApproval(a)
	if err != nil {
		t.Fatal(err)
	}
	q := guardianBegin{Approval: a, Claim: guardianClaim{Approval: hash}, Binary: r.binary}
	e, err := guardianFrame(q, 1, wc.KindReadiness, guardianEvent{PID: 123})
	if err != nil {
		t.Fatal(err)
	}
	var decoded guardianEvent
	if err := guardianDecode(e, q, 1, wc.KindReadiness, &decoded); err != nil {
		t.Fatal(err)
	}
	e.Payload = json.RawMessage(`{"pid":123,"error":"","extra":true}`)
	if err := guardianDecode(e, q, 1, wc.KindReadiness, &decoded); err == nil {
		t.Fatal("accepted unknown payload key")
	}
}

func TestGuardianReceiptMustMatchPublishedStart(t *testing.T) {
	for _, stage := range []string{"pending", "fenced", "owned", "done"} {
		t.Run(stage, func(t *testing.T) {
			s, a, r := guardianFixture(t)
			_, hash, err := freezeApproval(a)
			if err != nil {
				t.Fatal(err)
			}
			boot, err := hostBootIdentity()
			if err != nil {
				t.Fatal(err)
			}
			b, err := json.Marshal(guardianRecord{guardianClaim{hash, boot}, stage})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.path.WriteFile(guardianFile, b); err != nil {
				t.Fatal(err)
			}
			b, err = json.Marshal(physicalExit{physicalClaim{hash, boot}, 0})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.path.WriteFile(physicalExitFile, b); err != nil {
				t.Fatal(err)
			}
			started := contract.Outcome{JobID: a.Assignment.JobID, AttemptID: a.Assignment.AttemptID, ManifestHash: a.ManifestHash, ApprovalHash: hash, Version: 1, Kind: contract.Started, PID: 123}
			gone, err := r.Reconcile(context.Background(), Attempt{Version: contract.Version, Approval: a, Digest: hash, Started: &started})
			if err == nil || gone {
				t.Fatal("accepted cleanup inconsistent with start", gone, err)
			}
		})
	}
}
