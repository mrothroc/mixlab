//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

// ownedProcess retains the unreaped leader while signaling its process group.
// Once Wait releases the PID, all subsequent signal requests become no-ops.
// Approved workers must not detach descendants into other process groups.
type ownedProcess struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	reaped bool
	wait   chan error
	paused bool
	ending bool
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
	if signal == syscall.SIGTERM || signal == syscall.SIGKILL {
		p.ending = true
		if p.paused {
			// A stopped process cannot handle TERM. Never resume unmeasured
			// work merely to terminate it; KILL also wakes a stopped group.
			signal = syscall.SIGKILL
		}
	}
	err := syscall.Kill(-p.cmd.Process.Pid, signal)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

func (p *ownedProcess) suspend(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reaped || p.ending || p.paused || ctx.Err() != nil {
		return nil
	}
	if err := syscall.Kill(-p.cmd.Process.Pid, syscall.SIGSTOP); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return nil
		}
		return err
	}
	p.paused = true
	return nil
}

func (p *ownedProcess) resume(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reaped || p.ending || !p.paused || ctx.Err() != nil {
		return nil
	}
	if err := syscall.Kill(-p.cmd.Process.Pid, syscall.SIGCONT); err != nil {
		if errors.Is(err, syscall.ESRCH) {
			return nil
		}
		return err
	}
	p.paused = false
	return nil
}

func (p *ownedProcess) exitStatus() contract.ProcessExit {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.reaped || p.cmd.ProcessState == nil {
		return contract.ProcessExit{}
	}
	status, ok := p.cmd.ProcessState.Sys().(syscall.WaitStatus)
	if !ok {
		return contract.ProcessExit{}
	}
	if status.Exited() {
		return contract.ProcessExit{Kind: contract.ProcessExited, Code: status.ExitStatus()}
	}
	if status.Signaled() {
		return contract.ProcessExit{Kind: contract.ProcessSignaled, Signal: int(status.Signal())}
	}
	return contract.ProcessExit{}
}
