//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestProcessSampleDeadlineIdentity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := processSampleOutput(ctx, exec.CommandContext(ctx, "/bin/sleep", "30"))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("lost sampler deadline identity: %v", err)
	}
}

func TestProcessSampleNonContextFailure(t *testing.T) {
	ctx := context.Background()
	_, err := processSampleOutput(ctx, exec.CommandContext(ctx, "/bin/sh", "-c", "exit 7"))
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != 7 || errors.Is(err, context.Canceled) {
		t.Fatalf("lost independent process failure: %v", err)
	}
}

func TestResourceShutdownCancelsRunningSample(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	ready := filepath.Join(t.TempDir(), "sample-ready")
	sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
		cmd := exec.CommandContext(ctx, "/bin/sh", "-c", `printf ready > "$1"; exec sleep 30`, "sample", ready)
		kill := cmd.Cancel
		cmd.Cancel = func() error {
			err := kill()
			close(started)
			return err
		}
		_, err := processSampleOutput(ctx, cmd)
		return resourceUsage{}, err
	}
	stop := watchResources(parent, cancel, nil, "", contract.Limits{CPUSeconds: 1, MemoryBytes: 100, DiskBytes: 100}, sample)
	// Wait until the OS process exists before cancellation, rather than testing
	// only exec's easy context-already-canceled path.
	deadline := time.Now().Add(time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			_ = stop()
			t.Fatal("sample command did not start")
		}
		time.Sleep(time.Millisecond)
	}
	if err := stop(); err != nil {
		t.Fatalf("ordinary shutdown became a resource failure: %v", err)
	}
	select {
	case <-started:
	default:
		t.Fatal("test did not exercise cancellation of a running command")
	}
}

func TestResourceSampleDeadlineStillFailsClosed(t *testing.T) {
	parent, cancel := context.WithCancel(context.Background())
	defer cancel()
	sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
		probe, stop := context.WithTimeout(ctx, 50*time.Millisecond)
		defer stop()
		_, err := processSampleOutput(probe, exec.CommandContext(probe, "/bin/sleep", "30"))
		return resourceUsage{}, err
	}
	stop := watchResources(parent, cancel, nil, "", contract.Limits{CPUSeconds: 1, MemoryBytes: 100, DiskBytes: 100}, sample)
	select {
	case <-parent.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("sampler deadline did not cancel worker supervision")
	}
	if err := stop(); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("real deadline was suppressed: %v", err)
	}
}
