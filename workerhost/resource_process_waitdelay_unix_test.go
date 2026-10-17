//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"
)

type delayedDrainResult struct {
	output string
	err    error
	state  *os.ProcessState
}

// A descendant holds the real exec stdout pipe open after the command exits.
// Releasing fd 3 closes that pipe; EOF on fd 4 joins the holder without a sleep
// or signaling a numeric PID that may have been reaped. The caller's deadline
// also releases it, so a broken WaitDelay path cannot strand the fixture.
func delayedDrainSample(ctx context.Context, exitCode int, overflow bool) (result delayedDrainResult, fixtureErr error) {
	releaseRead, releaseWrite, err := os.Pipe()
	if err != nil {
		return result, err
	}
	defer func() { _ = releaseRead.Close(); _ = releaseWrite.Close() }()
	doneRead, doneWrite, err := os.Pipe()
	if err != nil {
		return result, err
	}
	defer func() { _ = doneRead.Close(); _ = doneWrite.Close() }()
	releaseOnCancel := context.AfterFunc(ctx, func() { _ = releaseWrite.Close() })
	defer releaseOnCancel()
	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", `
(read -r release <&3; printf done >&4) &
if [ "$2" = true ]; then
    /usr/bin/head -c 8388609 /dev/zero
else
    printf 'incomplete process sample\n'
fi
exit "$1"
`, "sample", strconv.Itoa(exitCode), strconv.FormatBool(overflow))
	cmd.Env = []string{"PATH=/usr/bin:/bin", "LC_ALL=C", "TZ=UTC"}
	cmd.ExtraFiles = []*os.File{releaseRead, doneWrite}
	result.output, result.err = processSampleOutput(ctx, cmd)
	result.state = cmd.ProcessState
	_ = releaseRead.Close()
	_ = doneWrite.Close()
	_ = releaseWrite.Close()
	if err := doneRead.SetReadDeadline(time.Now().Add(5 * time.Second)); err != nil {
		return result, err
	}
	done, err := io.ReadAll(doneRead)
	if err != nil {
		return result, fmt.Errorf("join output holder: %w", err)
	}
	if string(done) != "done" {
		return result, fmt.Errorf("output holder did not finish: result=%q", done)
	}
	return result, nil
}

func TestProcessSampleWaitDelayIsSlowObservation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r, err := delayedDrainSample(ctx, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	if r.state == nil || !r.state.Success() || ctx.Err() != nil {
		t.Fatalf("fixture did not exercise normal exit before probe deadline: state=%v ctx=%v", r.state, ctx.Err())
	}
	if !errors.Is(r.err, exec.ErrWaitDelay) || errors.Is(r.err, context.DeadlineExceeded) {
		t.Fatalf("lost actual drain timeout identity: %v", r.err)
	}
	if _, ok := r.err.(*processSampleTimeoutError); !ok || !resourceSampleSlow(r.err, nil) || !resourceSampleInterrupted(r.err) {
		t.Fatalf("confirmed drain delay did not enter bounded timeout tolerance: %v", r.err)
	}
	if r.output != "" {
		t.Fatalf("incomplete sample was returned as usable output: %q", r.output)
	}
}

func TestProcessSampleWaitDelayRetainsTerminalFailures(t *testing.T) {
	for _, tc := range []struct {
		name     string
		exitCode int
		overflow bool
	}{
		{"nonzero-exit", 7, false},
		{"output-bound", 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			r, err := delayedDrainSample(ctx, tc.exitCode, tc.overflow)
			if err != nil {
				t.Fatal(err)
			}
			if r.err == nil || r.output != "" || r.state == nil || r.state.ExitCode() != tc.exitCode || ctx.Err() != nil {
				t.Fatalf("fixture lost command/output failure: result=%+v ctx=%v", r, ctx.Err())
			}
			if resourceSampleSlow(r.err, context.DeadlineExceeded) || resourceSampleInterrupted(r.err) {
				t.Fatalf("real failure became a slow observation: %v", r.err)
			}
			var exit *exec.ExitError
			if tc.exitCode != 0 && (!errors.As(r.err, &exit) || exit.ExitCode() != tc.exitCode) {
				t.Fatalf("lost actual exit status: %v", r.err)
			}
			if tc.overflow && !strings.Contains(r.err.Error(), "process resource sample exceeds 8 MiB") {
				t.Fatalf("lost bounded-output I/O error: %v", r.err)
			}
		})
	}
}
