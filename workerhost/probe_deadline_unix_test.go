//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"testing/synctest"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func TestReadOnlyCommandDeadlinePolicy(t *testing.T) {
	for _, tc := range []struct {
		name   string
		parent time.Duration
		want   time.Duration
	}{
		{"unbounded-caller", 0, 2 * time.Minute},
		{"short-operation", time.Second, time.Second},
		{"admission-operation", 30 * time.Minute, 30 * time.Minute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				parent := context.Background()
				if tc.parent != 0 {
					var cancel context.CancelFunc
					parent, cancel = context.WithTimeout(parent, tc.parent)
					defer cancel()
				}
				started := time.Now()
				ctx, cancel := readOnlyCommandContext(parent)
				defer cancel()
				deadline, bounded := ctx.Deadline()
				if !bounded || deadline.Sub(started) != tc.want {
					t.Fatalf("probe changed caller/fallback bound: remaining=%s bounded=%v", deadline.Sub(started), bounded)
				}
				<-ctx.Done()
				if !errors.Is(ctx.Err(), context.DeadlineExceeded) || time.Since(started) != tc.want {
					t.Fatalf("probe deadline not enforced: elapsed=%s err=%v", time.Since(started), ctx.Err())
				}
			})
		})
	}
	parent, stop := context.WithCancel(context.Background())
	defer stop()
	ctx, cancel := readOnlyCommandContext(parent)
	cancel()
	if ctx.Err() != context.Canceled || parent.Err() != nil {
		t.Fatal("probe-local cancellation changed operation ownership")
	}
	ctx, cancel = readOnlyCommandContext(parent)
	defer cancel()
	stop()
	if ctx.Err() != context.Canceled {
		t.Fatal("probe detached from caller cancellation")
	}
}

// Use a hash-approved, fixed-mode executable with no numerical dependencies.
// Reply files are fixture-owned so the result can contain the executable's hash.
func preparationProbeFixture(t *testing.T, mode, behavior string) (*Supervisor, statehome.Path) {
	t.Helper()
	dir := privateDirectory(t)
	binary := filepath.Join(t.TempDir(), "approved-probe")
	script := `#!/bin/sh
test "$#" = 2 && test "$1" = -mode || exit 91
case "$2" in worker-probe|worker-learner-probe) ;; *) exit 92 ;; esac
behavior=$(/bin/cat behavior)
if [ "$behavior" = hang ]; then
    /bin/sleep 60 &
    printf '%s %s\n' "$$" "$!" > started
    wait
    exit 93
fi
printf '%s\n' "$$" > started
case "$behavior" in
    delay) /bin/sleep 16 ;;
    stdout-overflow) /usr/bin/head -c 131072 /dev/zero; exit 0 ;;
    stderr-overflow) /usr/bin/head -c 8192 /dev/zero >&2; exit 0 ;;
esac
exec /bin/cat "$2.reply"
`
	if err := os.WriteFile(binary, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	hash, err := workerjob.FileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(binary, hash)
	if err != nil {
		t.Fatal(err)
	}
	var reply any = workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "fixture", WorkerProtocol: wc.Version, OS: runtime.GOOS, Arch: runtime.GOARCH, DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	if mode == "worker-learner-probe" {
		h := strings.Repeat("a", 64)
		p := workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: hash, ConfigHash: h, ProgramHash: h, WeightLayoutHash: h, OptimizerHash: h, DType: "fp32", Learner: &workerprobe.LearnerPlan{
			Outer: run.Outer{Kind: "nesterov", LR: 1, Aggregation: "uniform", InnerSteps: 1, TotalSteps: 1, Accumulation: 1}, BatchTokens: 4, SequenceLength: 4, Precision: &precision.Policy{ComputeDType: "float32", Backend: "metal"},
		}}
		if err := p.Validate(); err != nil {
			t.Fatal(err)
		}
		reply = p
	}
	b, err := json.Marshal(reply)
	if err != nil {
		t.Fatal(err)
	}
	if err := dir.WriteFile(mode+".reply", b); err != nil {
		t.Fatal(err)
	}
	if err := dir.WriteFile("behavior", []byte(behavior)); err != nil {
		t.Fatal(err)
	}
	return s, dir
}

func callPreparationProbe(ctx context.Context, s *Supervisor, dir statehome.Path, mode string) error {
	if mode == "worker-learner-probe" {
		_, err := s.LearnerProbe(ctx, dir, []byte(`{"opaque":true}`))
		return err
	}
	_, err := s.Probe(ctx, dir)
	return err
}

func TestPreparationProbeOutlivesLegacyDeadline(t *testing.T) {
	for _, mode := range []string{"worker-learner-probe", "worker-probe"} {
		t.Run(mode, func(t *testing.T) {
			s, dir := preparationProbeFixture(t, mode, "delay")
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			started := time.Now()
			if err := callPreparationProbe(ctx, s, dir, mode); err != nil {
				t.Fatalf("caller-bounded initialization hit a hidden timeout: %v", err)
			}
			if time.Since(started) < 16*time.Second || ctx.Err() != nil {
				t.Fatalf("fixture did not outlast the old 15s limit: elapsed=%s ctx=%v", time.Since(started), ctx.Err())
			}
		})
	}
}

func TestPreparationProbeCancellationReapsGroup(t *testing.T) {
	for _, mode := range []string{"worker-learner-probe", "worker-probe"} {
		t.Run(mode, func(t *testing.T) {
			s, dir := preparationProbeFixture(t, mode, "hang")
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			done := make(chan error, 1)
			go func() { done <- callPreparationProbe(ctx, s, dir, mode) }()
			joined := false
			defer func() {
				cancel()
				if !joined {
					select {
					case <-done:
					case <-time.After(5 * time.Second):
						t.Error("probe did not join on cancellation")
					}
				}
			}()
			var pids []string
			for len(pids) != 2 {
				b, err := os.ReadFile(filepath.Join(dir.Dir(), "started"))
				if err != nil && !errors.Is(err, os.ErrNotExist) {
					t.Fatal(err)
				}
				pids = strings.Fields(string(b))
				select {
				case err := <-done:
					joined = true
					t.Fatalf("probe exited before cancellation: %v", err)
				case <-ctx.Done():
					t.Fatal("probe did not publish its live process group", ctx.Err())
				case <-time.After(10 * time.Millisecond):
				}
			}
			leader, err := strconv.Atoi(pids[0])
			if err != nil {
				t.Fatal(err)
			}
			child, err := strconv.Atoi(pids[1])
			if err != nil || syscall.Kill(child, 0) != nil {
				t.Fatal("probe descendant was not live", err)
			}
			cancel()
			select {
			case err := <-done:
				joined = true
				if !errors.Is(err, context.Canceled) || errors.Is(err, ErrReconciliationRequired) {
					t.Fatalf("probe cancellation lost identity or cleanup: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("probe ignored caller cancellation")
			}
			if err := syscall.Kill(leader, 0); err != syscall.ESRCH {
				t.Fatalf("probe leader was not reaped: %v", err)
			}
			if live, err := groupHasLiveMembers(leader); err != nil || live {
				t.Fatalf("probe left live descendants: live=%v err=%v", live, err)
			}
		})
	}
}

func TestPreparationProbeAlreadyCanceledDoesNotSpawn(t *testing.T) {
	s, dir := preparationProbeFixture(t, "worker-learner-probe", "hang")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := callPreparationProbe(ctx, s, dir, "worker-learner-probe"); !errors.Is(err, context.Canceled) {
		t.Fatal("lost caller cancellation", err)
	}
	if _, err := os.Stat(filepath.Join(dir.Dir(), "started")); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("canceled preparation spawned a probe", err)
	}
}

func TestPreparationProbeOutputBoundsRemainFatal(t *testing.T) {
	for _, mode := range []string{"worker-learner-probe", "worker-probe"} {
		for _, stream := range []string{"stdout-overflow", "stderr-overflow"} {
			t.Run(mode+"/"+stream, func(t *testing.T) {
				s, dir := preparationProbeFixture(t, mode, stream)
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				err := callPreparationProbe(ctx, s, dir, mode)
				if err == nil || !errors.Is(err, context.Canceled) || ctx.Err() != nil || errors.Is(err, ErrReconciliationRequired) {
					t.Fatalf("output overflow did not cancel only the bounded probe: %v parent=%v", err, ctx.Err())
				}
				if len(err.Error()) > 8192 {
					t.Fatal("probe diagnostic exceeded its bound")
				}
			})
		}
	}
}
