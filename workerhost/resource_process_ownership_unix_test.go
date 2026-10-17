//go:build darwin || linux

package workerhost

import (
	"bufio"
	"context"
	"os/exec"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestProcessSamplePlatformSelection(t *testing.T) {
	want := []string{"-axo", "pid=,pgid=,time=,rss=,lstart="}
	if runtime.GOOS == "darwin" {
		want = []string{"-g", "123", "-o", "pid=,pgid=,time=,rss=,lstart="}
	}
	if got := processSampleArgs(123); !reflect.DeepEqual(got, want) {
		t.Fatalf("platform process selection: got %q want %q", got, want)
	}
}

func TestProcessSampleIncludesOwnedGroupOnly(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	start := func(cmd *exec.Cmd) *ownedProcess {
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		p := ownProcess(cmd)
		t.Cleanup(func() {
			_ = p.signal(syscall.SIGKILL)
			<-p.wait
		})
		return p
	}
	foreign := start(exec.CommandContext(ctx, "/bin/sleep", "30"))
	cmd := exec.CommandContext(ctx, "/bin/sh", "-c", `sleep 30 & printf '%s\n' "$!"; wait`)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	owned := start(cmd)
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	child, err := strconv.Atoi(strings.TrimSpace(line))
	if err != nil {
		t.Fatal(err)
	}
	probe, stop := context.WithTimeout(ctx, resourceSampleTimeout)
	defer stop()
	usage, err := owned.sampleUsage(probe)
	if err != nil {
		t.Fatal(err)
	}
	want := map[int]bool{owned.cmd.Process.Pid: true, child: true}
	for _, process := range usage {
		pid, _, ok := strings.Cut(process.Identity, "/")
		n, err := strconv.Atoi(pid)
		if err != nil || !ok || !want[n] || n == foreign.cmd.Process.Pid {
			t.Fatalf("sample escaped owned process group: %+v", process)
		}
		delete(want, n)
	}
	if len(want) != 0 {
		t.Fatalf("missing owned processes %v in sample %+v", want, usage)
	}
}

func TestProcessSampleNeverUsesReapedPID(t *testing.T) {
	cmd := exec.Command("/usr/bin/true")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := ownProcess(cmd)
	if err := <-p.wait; err != nil {
		t.Fatal(err)
	}
	// An attempted numeric PID lookup after reaping must panic here rather
	// than passing accidentally because the PID has not yet been reused.
	p.cmd.Process = nil
	if usage, err := p.sampleUsage(context.Background()); err != nil || len(usage) != 0 {
		t.Fatalf("sampled reaped process: usage=%v err=%v", usage, err)
	}
}
