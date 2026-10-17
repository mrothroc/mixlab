//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

// Like the production node wrapper, this exposes only the Runner interface.
type exitRunnerWrapper struct{ Runner }

func TestDurableProcessExit(t *testing.T) {
	for _, guardian := range []bool{false, true} {
		name := "local"
		if guardian {
			name = "guardian"
		}
		for _, test := range []struct {
			selector string
			exit     contract.ProcessExit
		}{
			{"success", contract.ProcessExit{Kind: contract.ProcessExited}},
			{"exit-after-success", contract.ProcessExit{Kind: contract.ProcessExited, Code: 33}},
			{"exit-signal", contract.ProcessExit{Kind: contract.ProcessSignaled, Signal: int(syscall.SIGKILL)}},
		} {
			t.Run(name+"/"+test.selector, func(t *testing.T) {
				var s *AttemptStore
				var a contract.Approved
				var runner Runner
				if guardian {
					s, a, runner = guardianFixture(t)
				} else {
					s, a, runner = physicalFixture(t)
				}
				a.Assignment.DatasetSelector = test.selector
				runner = exitRunnerWrapper{runner}
				ctx := context.Background()
				o, err := s.Run(ctx, a, runner)
				if err != nil || o.Exit != test.exit || o.PID == 0 || !o.NoChild {
					t.Fatalf("durable OS outcome: %+v %v; want %+v", o, err, test.exit)
				}
				if (test.selector == "success") != (o.Kind == contract.Exited) {
					t.Fatalf("protocol/process outcome mismatch: %+v", o)
				}
				receipt, err := loadPhysicalExit(s.path)
				if err != nil || receipt.Exit != o.Exit || receipt.PID != o.PID {
					t.Fatalf("physical receipt: %+v %v", receipt, err)
				}
				// Model a crash after the receipt but before terminal publication.
				old, state, err := s.load()
				if err != nil {
					t.Fatal(err)
				}
				state.Terminal = nil
				if err := s.save(old, state); err != nil {
					t.Fatal(err)
				}
				if guardian {
					b, record, err := loadGuardian(s.path)
					if err != nil {
						t.Fatal(err)
					}
					record.State = "owned"
					next, err := json.Marshal(record)
					if err != nil {
						t.Fatal(err)
					}
					if err := s.path.CompareAndSwap(guardianFile, b, next); err != nil {
						t.Fatal(err)
					}
				}
				reopened, err := NewAttemptStore(s.path)
				if err != nil {
					t.Fatal(err)
				}
				recovered, err := reopened.Reconcile(ctx, runner)
				if err != nil || recovered.Exit != o.Exit || recovered.PID != o.PID || recovered.Kind != contract.Failed {
					t.Fatalf("reconciliation lost OS evidence: %+v %v", recovered, err)
				}
				if replay, err := reopened.Run(ctx, a, runner); err != nil || replay != recovered {
					t.Fatalf("terminal replay changed evidence: %+v %v", replay, err)
				}
			})
		}
	}
}

func TestProcessExitNoSpawnAndUnknown(t *testing.T) {
	t.Run("no_spawn", func(t *testing.T) {
		s, a, r := physicalFixture(t)
		r.supervisor.build = strings.Repeat("f", 64)
		o, err := s.Run(context.Background(), a, r)
		if err != nil || o.Kind != contract.Failed || o.PID != 0 || o.Exit != (contract.ProcessExit{Kind: contract.ProcessNotStarted}) {
			t.Fatalf("no spawn: %+v %v", o, err)
		}
	})
	t.Run("protocol_error_without_wait_evidence", func(t *testing.T) {
		s, a := attemptFixture(t)
		r := fakeRunner{run: func(_ context.Context, _ contract.Approved, started func(int) error) error {
			if err := started(123); err != nil {
				return err
			}
			return errors.New("EOF")
		}}
		o, err := s.Run(context.Background(), a, r)
		if err != nil || o.Kind != contract.Failed || o.Exit != (contract.ProcessExit{}) {
			t.Fatalf("protocol error invented OS status: %+v %v", o, err)
		}
	})
}

func TestProcessExitBeforeStartedPublication(t *testing.T) {
	s, a, physical := physicalFixture(t)
	r := fakeRunner{run: func(ctx context.Context, a contract.Approved, _ func(int) error) error {
		return physical.Run(ctx, a, func(int) error { return errors.New("started publication unavailable") })
	}, reconcile: physical.Reconcile}
	o, err := s.Run(context.Background(), a, r)
	if err != nil || o.PID == 0 || o.Version != 1 || o.Exit.Kind == "" || o.Exit.Kind == contract.ProcessNotStarted {
		t.Fatalf("startup physical evidence lost: %+v %v", o, err)
	}
	old, state, err := s.load()
	if err != nil || state.Started != nil {
		t.Fatalf("invented Started history: %+v %v", state, err)
	}
	state.Terminal = nil
	if err := s.save(old, state); err != nil {
		t.Fatal(err)
	}
	recovered, err := s.Reconcile(context.Background(), r)
	if err != nil || recovered.PID != o.PID || recovered.Exit != o.Exit || recovered.Version != 1 {
		t.Fatalf("startup reconciliation lost evidence: %+v %v", recovered, err)
	}
}

func TestLegacyPhysicalExitCompatibility(t *testing.T) {
	s, a, r := physicalFixture(t)
	o, err := s.Run(context.Background(), a, r)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := loadPhysicalExit(s.path)
	if err != nil {
		t.Fatal(err)
	}
	// This is the exact pre-diagnostics receipt schema, not a new-schema round trip.
	legacy, err := json.Marshal(struct {
		Claim physicalClaim `json:"claim"`
		PID   int           `json:"pid"`
	}{receipt.Claim, receipt.PID})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.path.WriteFile(physicalExitFile, legacy); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadPhysicalExit(s.path)
	if err != nil || loaded.Exit != (contract.ProcessExit{}) {
		t.Fatalf("legacy receipt rejected or invented exit zero: %+v %v", loaded, err)
	}
	b, err := json.Marshal(loaded)
	if err != nil || !bytes.Equal(b, legacy) {
		t.Fatalf("legacy canonical bytes changed: %s %v", b, err)
	}
	old, state, err := s.load()
	if err != nil {
		t.Fatal(err)
	}
	state.Terminal = nil
	if err := s.save(old, state); err != nil {
		t.Fatal(err)
	}
	recovered, err := s.Reconcile(context.Background(), r)
	if err != nil || recovered.PID != o.PID || recovered.Exit != (contract.ProcessExit{}) {
		t.Fatalf("legacy reconciliation: %+v %v", recovered, err)
	}
	// Reboot proves cleanup, not the old process's exit code or signal.
	old, state, err = s.load()
	if err != nil {
		t.Fatal(err)
	}
	state.Terminal = nil
	if err := s.save(old, state); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(s.path.Dir(), physicalExitFile)); err != nil {
		t.Fatal(err)
	}
	r.boot = func() (string, error) { return "11111111-1111-1111-1111-111111111111", nil }
	recovered, err = s.Reconcile(context.Background(), r)
	if err != nil || recovered.Exit != (contract.ProcessExit{}) {
		t.Fatalf("reboot invented exit status: %+v %v", recovered, err)
	}
}
