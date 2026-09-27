//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"time"

	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// ServeGuardian is an internal composition entry point. It accepts a single
// inherited Unix socket, never listens remotely, and holds no trust credentials.
// The socket's lifetime is the agent's cancellation capability.
func ServeGuardian(file *os.File) error {
	if file == nil {
		return fmt.Errorf("inherited guardian descriptor required")
	}
	c, err := net.FileConn(file)
	_ = file.Close()
	if err != nil {
		return err
	}
	defer func() { _ = c.Close() }()
	if _, ok := c.(*net.UnixConn); !ok {
		return fmt.Errorf("guardian requires an inherited Unix socket")
	}
	if err := c.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
		return err
	}
	e, err := wc.ReadFrame(c, wc.MaxFrameBytes)
	if err != nil {
		return err
	}
	var q guardianBegin
	// Decode once to obtain the expected immutable frame binding, then enforce
	// exact schema, envelope and self-build before any process can be launched.
	if err := decodeGuardianBegin(e, &q); err != nil {
		return err
	}
	dir, err := q.directory()
	if err != nil {
		return err
	}
	s, err := New(q.Binary, q.Approval.Assignment.BuildID)
	if err != nil {
		return err
	}
	r, err := NewLocalRunner(s, dir, q.Startup, q.Grace)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(q.Approval.Assignment.RuntimeSeconds)*time.Second)
	defer cancel()
	if err := transitionGuardian(ctx, dir, q.Claim, "pending", "owned"); err != nil {
		return err
	}
	// One reader owns the agent channel, including during spawn/start publication.
	// EOF or any unexpected command cancels physical supervision, even if the
	// worker is hung in a native call and cannot honor cooperative cancellation.
	if err := c.SetReadDeadline(time.Time{}); err != nil {
		return err
	}
	proceed := make(chan error, 1)
	readerDone := make(chan struct{})
	go func() {
		defer close(readerDone)
		msg, err := wc.ReadFrame(c, wc.MaxFrameBytes)
		if err == nil {
			var ack guardianEvent
			err = guardianDecode(msg, q, 2, wc.KindReadiness, &ack)
			if ack != (guardianEvent{}) {
				err = fmt.Errorf("invalid guardian acknowledgment")
			}
		}
		proceed <- err
		if err == nil {
			_, _ = wc.ReadFrame(c, wc.MaxFrameBytes)
		}
		cancel()
	}()
	defer func() { _ = c.Close(); <-readerDone }()
	sequence := uint64(1)
	runErr := r.Run(ctx, q.Approval, func(pid int) error {
		if err := guardianSend(c, q, sequence, wc.KindReadiness, guardianEvent{PID: pid}); err != nil {
			return err
		}
		sequence++
		deadline := time.NewTimer(q.Startup)
		defer deadline.Stop()
		select {
		case err := <-proceed:
			return err
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("guardian start acknowledgment expired")
		}
	})
	if !errors.Is(runErr, ErrReconciliationRequired) {
		cleanupCtx, done := context.WithTimeout(context.Background(), 5*time.Second)
		// Do not infer cleanup solely from Run's error class: an early lock
		// failure might have occurred before it examined existing ownership.
		gone, checkErr := r.Reconcile(cleanupCtx, Attempt{Version: contract.Version, Approval: q.Approval, Digest: q.Claim.Approval})
		if checkErr != nil || !gone {
			err = errors.Join(checkErr, ErrReconciliationRequired)
		} else {
			err = transitionGuardian(cleanupCtx, dir, q.Claim, "owned", "done")
		}
		done()
		if err != nil {
			runErr = errors.Join(runErr, err, ErrReconciliationRequired)
		}
	}
	event := guardianEvent{}
	if runErr != nil {
		event.Error = runErr.Error()
		if len(event.Error) > 4096 {
			event.Error = "guardian attempt failed; see local attempt state"
		}
	}
	_ = guardianSend(c, q, sequence, wc.KindTerminalOutcome, event)
	return runErr
}

func decodeGuardianBegin(e wc.Envelope, q *guardianBegin) error {
	// guardianDecode validates the complete canonical payload after this decode.
	if err := workerjob.DecodePayload(e.Payload, q); err != nil {
		return err
	}
	if err := guardianDecode(e, *q, 1, wc.KindAssignment, q); err != nil {
		return err
	}
	_, hash, err := freezeApproval(q.Approval)
	if err != nil {
		return err
	}
	boot, err := hostBootIdentity()
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	build, err := workerjob.FileDigest(exe)
	if err != nil || build != q.GuardianBuild || q.Claim.Approval != hash || q.Claim.Boot != boot {
		return fmt.Errorf("guardian build/approval/boot binding mismatch")
	}
	return nil
}
