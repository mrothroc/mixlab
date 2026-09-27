//go:build darwin || linux

package workerhost

import (
	"os/exec"
	"syscall"
	"testing"
)

func TestExitObservationRetainsChildUntilReaping(t *testing.T) {
	for range 20 {
		cmd := exec.Command("/usr/bin/true")
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		err := observeChildExit(cmd.Process.Pid)
		probe := syscall.Kill(cmd.Process.Pid, 0)
		waitErr := cmd.Wait()
		if err != nil || probe != nil || waitErr != nil {
			t.Fatalf("exit observation reaped/lost our child: %v %v %v", err, probe, waitErr)
		}
	}
}

func TestOwnedProcessNeverSignalsAfterReap(t *testing.T) {
	cmd := exec.Command("/usr/bin/true")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	p := ownProcess(cmd)
	if err := <-p.wait; err != nil {
		t.Fatal(err)
	}
	// Remove the process pointer: any attempt to address a numeric PID after
	// reaping now panics instead of silently passing because it was not reused.
	p.cmd.Process = nil
	if err := p.signal(syscall.SIGKILL); err != nil {
		t.Fatal(err)
	}
}
