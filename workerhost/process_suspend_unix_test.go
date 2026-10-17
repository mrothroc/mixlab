//go:build darwin || linux

package workerhost

import (
	"bufio"
	"context"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func startSuspendTestProcess(t *testing.T, cmd *exec.Cmd) *ownedProcess {
	t.Helper()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := ownProcess(cmd)
	t.Cleanup(func() {
		_ = p.signal(syscall.SIGKILL)
		p.mu.Lock()
		reaped := p.reaped
		p.mu.Unlock()
		if !reaped {
			select {
			case <-p.wait:
			case <-time.After(5 * time.Second):
				t.Error("test group cleanup did not reap")
			}
		}
	})
	return p
}

func awaitGroupStopped(t *testing.T, pids []int, stopped bool) {
	t.Helper()
	ids := make([]string, len(pids))
	for i, pid := range pids {
		ids[i] = strconv.Itoa(pid)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var last string
	for ctx.Err() == nil {
		out, err := exec.CommandContext(ctx, "/bin/ps", "-p", strings.Join(ids, ","), "-o", "pid=,state=").Output()
		if err != nil {
			t.Fatalf("read group state: %v", err)
		}
		last = string(out)
		lines := strings.Split(strings.TrimSpace(last), "\n")
		match := len(lines) == len(pids)
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) != 2 || strings.HasPrefix(fields[1], "T") != stopped {
				match = false
			}
		}
		if match {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("processes %v stopped=%v not observed: %s", pids, stopped, last)
}

func TestOwnedProcessGroupSuspendResumeAndCleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	foreign := startSuspendTestProcess(t, exec.Command("/bin/sleep", "60"))
	cmd := exec.Command("/bin/sh", "-c", `sleep 60 & printf '%s\n' "$!"; wait`)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	p := startSuspendTestProcess(t, cmd)
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	child, err := strconv.Atoi(strings.TrimSpace(line))
	if err != nil {
		t.Fatal(err)
	}
	pids := []int{cmd.Process.Pid, child}
	if err := p.suspend(ctx); err != nil {
		t.Fatal(err)
	}
	awaitGroupStopped(t, pids, true)
	awaitGroupStopped(t, []int{foreign.cmd.Process.Pid}, false)
	probe, stop := context.WithTimeout(ctx, resourceSampleTimeout)
	usage, err := p.sampleUsage(probe)
	stop()
	if err != nil || len(usage) != 2 {
		t.Fatalf("cannot measure suspended owned group: usage=%v err=%v", usage, err)
	}
	var meter resourceMeter
	if err := meter.check(resourceUsage{Processes: usage}, contract.Limits{CPUSeconds: 60, MemoryBytes: 1 << 30}); err != nil {
		t.Fatal(err)
	}
	if err := p.resume(ctx); err != nil {
		t.Fatal(err)
	}
	awaitGroupStopped(t, pids, false)
	if err := p.suspend(ctx); err != nil {
		t.Fatal(err)
	}
	awaitGroupStopped(t, pids, true)
	cancel()
	if err := p.resume(ctx); err != nil {
		t.Fatal(err)
	}
	awaitGroupStopped(t, pids, true)
	if err := p.signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	// Cleanup must not need CONT to deliver TERM to a stopped learner.
	select {
	case <-p.wait:
	case <-time.After(5 * time.Second):
		t.Fatal("suspended group did not terminate promptly")
	}
	if got := p.exitStatus(); got.Kind != contract.ProcessSignaled || got.Signal != int(syscall.SIGKILL) {
		t.Fatalf("suspended cleanup did not use KILL: %+v", got)
	}
	awaitGroupStopped(t, []int{foreign.cmd.Process.Pid}, false)
	// Reaping or cleanup must win over any late pause/resume callback.
	p.cmd.Process = nil
	if err := p.suspend(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := p.resume(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestOwnedProcessEndingCannotResume(t *testing.T) {
	p := &ownedProcess{paused: true, ending: true}
	if err := p.resume(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := p.suspend(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !p.paused {
		t.Fatal("ending process was resumed")
	}
}

func TestLearnerResourceMonitorSuspendsRealGroup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := startSuspendTestProcess(t, exec.Command("/bin/sleep", "60"))
	calls := 0
	started := make(chan struct{})
	release := make(chan struct{})
	sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
		calls++
		if calls <= 2 {
			// Inject confirmed sample deadlines, without burning CPU or waiting
			// for two real ps timeouts on the developer's host.
			return resourceUsage{}, &processSampleTimeoutError{cause: context.DeadlineExceeded}
		}
		close(started)
		<-release // Model disk I/O that ignores cancellation while paused.
		return resourceUsage{}, nil
	}
	stop := watchLearnerResources(ctx, cancel, p, "", contract.Limits{}, sample)
	defer func() {
		cancel()
		close(release)
		if err := stop(); err != nil {
			t.Errorf("cancel suspended monitor: %v", err)
		}
	}()
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("monitor did not enter suspended sampling")
	}
	awaitGroupStopped(t, []int{p.cmd.Process.Pid}, true)
	if ctx.Err() != nil {
		t.Fatal("sampling delays canceled real learner")
	}
	cancel()
	if err := p.signal(syscall.SIGTERM); err != nil {
		t.Fatal(err)
	}
	select {
	case <-p.wait:
	case <-time.After(5 * time.Second):
		t.Fatal("cleanup waited for blocked sampler while suspended")
	}
}
