//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// GuardianRunner uses the approved cluster executable as a per-attempt local
// supervisor. The helper survives agent death long enough to stop/reap its own
// child and publish cleanup. No worker PID signal is issued by a restarted agent.
type GuardianRunner struct {
	local         *LocalRunner
	binary, build string
}

func NewGuardianRunner(local *LocalRunner, binary, build string) (*GuardianRunner, error) {
	if local == nil || !filepath.IsAbs(binary) {
		return nil, fmt.Errorf("local runner and approved absolute guardian binary required")
	}
	hash, err := workerjob.FileDigest(binary)
	if err != nil || hash != build {
		return nil, fmt.Errorf("guardian executable digest mismatch")
	}
	return &GuardianRunner{local, binary, build}, nil
}

func (r *GuardianRunner) Run(parent context.Context, a contract.Approved, started func(int) error) (err error) {
	a, hash, err := freezeApproval(a)
	if err != nil {
		return err
	}
	if started == nil {
		return fmt.Errorf("durable start callback required")
	}
	if err := parent.Err(); err != nil {
		return err
	}
	current, err := workerjob.FileDigest(r.binary)
	if err != nil || current != r.build {
		return fmt.Errorf("approved guardian executable changed")
	}
	boot, err := hostBootIdentity()
	if err != nil {
		return err
	}
	q := guardianBegin{Approval: a, Directory: r.local.directory.Dir(), Binary: r.local.supervisor.binary,
		GuardianBuild: r.build, Startup: r.local.startup, Grace: r.local.grace, Claim: guardianClaim{hash, boot}}
	b, err := json.Marshal(guardianRecord{q.Claim, "pending"})
	if err != nil {
		return err
	}
	if err := r.local.directory.CompareAndSwap(guardianFile, nil, b); err != nil {
		return errors.Join(ErrReconciliationRequired, err)
	}
	// On every error, ask the durable owner before claiming NoChild. A pending
	// helper can be atomically fenced; an owned helper must finish cleanup.
	startedPID := 0
	defer func() {
		ctx, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		gone, checkErr := reconcileGuardian(ctx, r.local.directory, hash, startedPID)
		if checkErr != nil || !gone {
			err = errors.Join(err, checkErr, ErrReconciliationRequired)
		}
	}()
	c, child, err := guardianSocketPair()
	if err != nil {
		return err
	}
	defer func() { _ = c.Close(); _ = child.Close() }()
	ctx, cancel := context.WithTimeout(parent, time.Duration(a.Assignment.RuntimeSeconds)*time.Second+q.Startup+10*time.Second)
	defer cancel()
	stopClose := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stopClose()
	cmd := exec.Command(r.binary, guardianCommand)
	cmd.Dir = q.Directory
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + q.Directory, "TMPDIR=" + q.Directory}
	cmd.ExtraFiles = []*os.File{child}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = time.Second
	diagnostic := &probeBuffer{limit: 4096, cancel: func() { _ = c.Close() }}
	cmd.Stderr = diagnostic
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	_ = child.Close()
	helper := ownProcess(cmd)
	defer func() {
		_ = c.Close()
		select {
		case exitErr := <-helper.wait:
			if exitErr != nil {
				err = errors.Join(err, fmt.Errorf("guardian exit: %w", exitErr))
			}
		case <-time.After(q.Grace + resourceSampleTimeout + 10*time.Second):
			// Keep the native-free guardian alive: killing it would discard the
			// only owner able to reap a worker. EOF has requested cancellation;
			// its existing Wait goroutine still reaps eventual helper exit.
			err = errors.Join(err, ErrReconciliationRequired)
		}
	}()
	if err := guardianSend(c, q, 1, wc.KindAssignment, q); err != nil {
		return err
	}
	if err := c.SetReadDeadline(time.Now().Add(q.Startup + 10*time.Second)); err != nil {
		return err
	}
	e, err := wc.ReadFrame(c, wc.MaxFrameBytes)
	if err != nil {
		return err
	}
	sequence := uint64(1)
	if e.Kind == wc.KindReadiness {
		var ready guardianEvent
		if err := guardianDecode(e, q, sequence, wc.KindReadiness, &ready); err != nil {
			return err
		}
		if ready.PID < 1 || ready.Error != "" {
			return fmt.Errorf("invalid guardian start event")
		}
		startedPID = ready.PID
		if err := started(ready.PID); err != nil {
			return err
		}
		if err := guardianSend(c, q, 2, wc.KindReadiness, guardianEvent{}); err != nil {
			return err
		}
		sequence++
		if err := c.SetReadDeadline(time.Now().Add(time.Duration(a.Assignment.RuntimeSeconds)*time.Second + 10*time.Second)); err != nil {
			return err
		}
		e, err = wc.ReadFrame(c, wc.MaxFrameBytes)
		if err != nil {
			return err
		}
	}
	var terminal guardianEvent
	if err := guardianDecode(e, q, sequence, wc.KindTerminalOutcome, &terminal); err != nil {
		return err
	}
	if terminal.PID != 0 || len(terminal.Error) > 4096 {
		return fmt.Errorf("invalid guardian terminal event")
	}
	if terminal.Error != "" {
		return errors.New(terminal.Error)
	}
	return ctx.Err()
}

func (r *GuardianRunner) Reconcile(ctx context.Context, a Attempt) (bool, error) {
	if err := a.validate(); err != nil {
		return false, err
	}
	pid := 0
	if a.Started != nil {
		pid = a.Started.PID
	}
	return reconcileGuardian(ctx, r.local.directory, a.Digest, pid)
}

var _ Runner = (*GuardianRunner)(nil)
