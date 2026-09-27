//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

type fakeRunner struct {
	run       func(context.Context, contract.Approved, func(int) error) error
	reconcile func(context.Context, Attempt) (bool, error)
}

func (f fakeRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	return f.run(ctx, a, started)
}
func (f fakeRunner) Reconcile(ctx context.Context, a Attempt) (bool, error) {
	return f.reconcile(ctx, a)
}

func attemptFixture(t *testing.T) (*AttemptStore, contract.Approved) {
	t.Helper()
	_, p := plan(t, "success")
	s, err := NewAttemptStore(p.Directory)
	if err != nil {
		t.Fatal(err)
	}
	a := contract.Approved{Version: contract.Version, ManifestHash: strings.Repeat("a", 64), TransportHash: strings.Repeat("b", 64), Assignment: p.Assignment, Limits: contract.Limits{CPUSeconds: 60, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 4096}}
	return s, a
}

func TestAttemptExactlyOneLauncherAndImmutableOutcomes(t *testing.T) {
	s, a := attemptFixture(t)
	var launched atomic.Int32
	r := fakeRunner{run: func(_ context.Context, got contract.Approved, started func(int) error) error {
		launched.Add(1)
		status, err := s.Status()
		if err != nil || status.Started != nil || status.Approval.Assignment.JobID != got.Assignment.JobID {
			t.Error("missing durable pre-spawn intent", err)
		}
		if err := started(123); err != nil {
			return err
		}
		status, err = s.Status()
		if err != nil || status.Started == nil || status.Started.PID != 123 {
			t.Error("missing durable started outcome", err)
		}
		return nil
	}}
	const n = 8
	var wg sync.WaitGroup
	outcomes := make(chan contract.Outcome, n)
	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			o, e := s.Run(context.Background(), a, r)
			if e != nil {
				t.Error(e)
			}
			outcomes <- o
		}()
	}
	wg.Wait()
	close(outcomes)
	var first contract.Outcome
	for o := range outcomes {
		if first.Version == 0 {
			first = o
		}
		if o != first {
			t.Fatal("retry changed outcome")
		}
	}
	if launched.Load() != 1 || first.Kind != contract.Exited || first.Version != 2 || !first.NoChild {
		t.Fatal("bad execution outcome", launched.Load(), first)
	}
	s2, err := NewAttemptStore(s.path)
	if err != nil {
		t.Fatal(err)
	}
	o, err := s2.Run(context.Background(), a, r)
	if err != nil || o != first || launched.Load() != 1 {
		t.Fatal("restart lost terminal", o, err)
	}
	a.TransportHash = strings.Repeat("c", 64)
	if _, err := s.Run(context.Background(), a, r); err == nil {
		t.Fatal("changed approval accepted")
	}
}

func TestApprovedUnclaimedAttemptCanBeFenced(t *testing.T) {
	s, a := attemptFixture(t)
	o, err := s.FenceApproved(context.Background(), a)
	if err != nil || o.ApprovalHash == "" || !o.NoChild || o.Kind != contract.Canceled {
		t.Fatal(o, err)
	}
	again, err := s.FenceApproved(context.Background(), a)
	if err != nil || again != o {
		t.Fatal(again, err)
	}
	got, err := s.Run(context.Background(), a, fakeRunner{run: func(context.Context, contract.Approved, func(int) error) error {
		t.Error("launched fenced approval")
		return nil
	}})
	if err != nil || got != o {
		t.Fatal(got, err)
	}
}

func TestAttemptUnpublishedInitializationCanBeCanceledButNeverRestarted(t *testing.T) {
	for _, point := range []string{"claim", "journal"} {
		t.Run(point, func(t *testing.T) {
			s, a := attemptFixture(t)
			a, hash, err := freezeApproval(a)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.path.CompareAndSwap(attemptClaim, nil, []byte(hash)); err != nil {
				t.Fatal(err)
			}
			if point == "journal" {
				if err := s.save(nil, Attempt{Version: contract.Version, Approval: a, Digest: hash}); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := s.Status(); err == nil {
				t.Fatal("unpublished attempt visible")
			}
			r := fakeRunner{run: func(context.Context, contract.Approved, func(int) error) error {
				t.Error("unpublished attempt resumed")
				return nil
			}}
			if _, err := s.Run(context.Background(), a, r); err == nil {
				t.Fatal("unpublished attempt restarted")
			}
			o, err := s.FenceApproved(context.Background(), a)
			if err != nil || !o.NoChild || o.Kind != contract.Canceled {
				t.Fatal(o, err)
			}
			if _, err := s.Status(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAttemptUncertainTeardownRemainsNonterminal(t *testing.T) {
	s, a := attemptFixture(t)
	r := fakeRunner{run: func(_ context.Context, _ contract.Approved, started func(int) error) error {
		if err := started(123); err != nil {
			return err
		}
		return ErrReconciliationRequired
	}}
	if _, err := s.Run(context.Background(), a, r); !errors.Is(err, ErrReconciliationRequired) {
		t.Fatal(err)
	}
	status, err := s.Status()
	if err != nil || status.Started == nil || status.Terminal != nil {
		t.Fatal(status, err)
	}
}

func TestAttemptCrashGapsNeverRelaunch(t *testing.T) {
	for _, point := range []string{"intent", "started", "terminal-publication"} {
		t.Run(point, func(t *testing.T) {
			s, a := attemptFixture(t)
			a, hash, err := freezeApproval(a)
			if err != nil {
				t.Fatal(err)
			}
			r := Attempt{Version: contract.Version, Approval: a, Digest: hash}
			if point != "intent" {
				o := contract.Outcome{JobID: a.Assignment.JobID, AttemptID: a.Assignment.AttemptID, ManifestHash: a.ManifestHash, ApprovalHash: hash, Kind: contract.Started, Version: 1, PID: 123}
				r.Started = &o
			}
			if err := s.path.CompareAndSwap(attemptClaim, nil, []byte(hash)); err != nil {
				t.Fatal(err)
			}
			if err := s.save(nil, r); err != nil {
				t.Fatal(err)
			}
			if err := s.path.CompareAndSwap(attemptReady, nil, []byte(hash)); err != nil {
				t.Fatal(err)
			}
			var calls int
			runner := fakeRunner{run: func(context.Context, contract.Approved, func(int) error) error { calls++; return nil }, reconcile: func(context.Context, Attempt) (bool, error) { return false, nil }}
			if _, err := s.Run(context.Background(), a, runner); !errors.Is(err, ErrReconciliationRequired) || calls != 0 {
				t.Fatal("duplicate spawn allowed", err, calls)
			}
			if _, err := s.Reconcile(context.Background(), runner); !errors.Is(err, ErrReconciliationRequired) {
				t.Fatal("uncertain child reported terminal", err)
			}
			runner.reconcile = func(context.Context, Attempt) (bool, error) { return true, nil }
			o, err := s.Reconcile(context.Background(), runner)
			if err != nil || o.Kind != contract.Failed || !o.NoChild {
				t.Fatal(o, err)
			}
			got, err := s.Run(context.Background(), a, runner)
			if err != nil || got != o || calls != 0 {
				t.Fatal("reconciled attempt restarted", got, err)
			}
		})
	}
}

func TestAttemptFailureCancellationAndMissingHistory(t *testing.T) {
	for _, mode := range []string{"launch-error", "started-write-error", "canceled", "missing-started"} {
		t.Run(mode, func(t *testing.T) {
			s, a := attemptFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := fakeRunner{run: func(_ context.Context, _ contract.Approved, started func(int) error) error {
				switch mode {
				case "launch-error":
					return errors.New("spawn failed")
				case "started-write-error":
					return started(0)
				case "canceled":
					if err := started(123); err != nil {
						return err
					}
					cancel()
					return ctx.Err()
				default:
					return nil
				}
			}}
			o, err := s.Run(ctx, a, r)
			if err != nil {
				t.Fatal(err)
			}
			want := contract.Failed
			if mode == "canceled" {
				want = contract.Canceled
			}
			if o.Kind != want || !o.NoChild || o.Error == "" {
				t.Fatal("false success", o)
			}
			if err := os.Remove(filepath.Join(s.path.Dir(), attemptFile)); err != nil {
				t.Fatal(err)
			}
			if _, err := s.Run(context.Background(), a, r); err == nil {
				t.Fatal("missing history recreated")
			}
		})
	}
}

func TestPreparationFencePreventsStart(t *testing.T) {
	s, a := attemptFixture(t)
	f := contract.Fence{JobID: a.Assignment.JobID, AttemptID: a.Assignment.AttemptID, ManifestHash: a.ManifestHash}
	o, err := s.FencePreparation(context.Background(), f)
	if err != nil {
		t.Fatal(err)
	}
	if !o.NoChild || o.Kind != contract.Canceled || o.ApprovalHash != "" {
		t.Fatal(o)
	}
	got, err := s.FencePreparation(context.Background(), f)
	if err != nil || got != o {
		t.Fatal("fence retry", got, err)
	}
	r := fakeRunner{run: func(context.Context, contract.Approved, func(int) error) error {
		t.Fatal("fenced job launched")
		return nil
	}}
	if _, err := s.Run(context.Background(), a, r); err == nil {
		t.Fatal("fenced job start accepted")
	}
	f.AttemptID = "other"
	if _, err := s.FencePreparation(context.Background(), f); err == nil {
		t.Fatal("changed fence accepted")
	}
}
