//go:build darwin || linux

package workerhost

import (
	"errors"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// ownedProcess retains the unreaped leader while signaling its process group.
// Once Wait releases the PID, all subsequent signal requests become no-ops.
// Approved workers must not detach descendants into other process groups.
type ownedProcess struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	reaped bool
	wait   chan error
}

func ownProcess(cmd *exec.Cmd) *ownedProcess {
	p := &ownedProcess{cmd: cmd, wait: make(chan error, 1)}
	go func() {
		observeErr := observeChildExit(cmd.Process.Pid)
		p.mu.Lock()
		// The leader is still ours, including when it is a zombie. Killing the
		// group before reaping cannot address an unrelated reused process group.
		killErr := terminateOwnedGroup(cmd.Process.Pid)
		waitErr := cmd.Wait()
		p.reaped = true
		p.mu.Unlock()
		if killErr != nil {
			waitErr = errors.Join(waitErr, killErr, ErrReconciliationRequired)
		}
		p.wait <- errors.Join(observeErr, waitErr)
	}()
	return p
}

func terminateOwnedGroup(pid int) error {
	// Darwin can return EPERM for an all-zombie group. Corroborate the actual
	// process table instead of treating either EPERM or successful kill as proof.
	signalErr := syscall.Kill(-pid, syscall.SIGKILL)
	deadline := time.Now().Add(time.Second)
	for {
		live, err := groupHasLiveMembers(pid)
		if err != nil {
			return errors.Join(signalErr, err)
		}
		if !live {
			return nil
		}
		if time.Now().After(deadline) {
			return errors.Join(signalErr, ErrReconciliationRequired)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func (p *ownedProcess) signal(signal syscall.Signal) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reaped {
		return nil
	}
	err := syscall.Kill(-p.cmd.Process.Pid, signal)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}
