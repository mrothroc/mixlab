//go:build darwin || linux

// Package workerhost supervises one approved local worker process. Approval,
// durable leases, job journals and cohort-wide compensation belong to callers.
// This package deliberately never links the trainer or GPU runtime.
package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// Supervisor is constructed by the administrator's approved-build registry,
// not from a submitted job. No executable path, argv or environment is accepted
// in LaunchPlan. Installed executable and library directories must be immutable
// for the attempt on trusted administrator-controlled hosts.
type Supervisor struct {
	binary, build string
	sample        resourceSampler // nil selects the platform sampler; tests may inject faults.
}

func New(binary, sha256 string) (*Supervisor, error) {
	if !filepath.IsAbs(binary) {
		return nil, fmt.Errorf("approved worker binary must be absolute")
	}
	id, err := workerjob.FileDigest(binary)
	if err != nil || id != sha256 {
		return nil, fmt.Errorf("approved worker build digest mismatch")
	}
	return &Supervisor{binary: binary, build: id}, nil
}

type LaunchPlan struct {
	Assignment     workerjob.Assignment
	Directory      statehome.Path // fresh private directory allocated by hosting
	StartupTimeout time.Duration
	ShutdownGrace  time.Duration
	Limits         contract.Limits
	Started        func(int) error // local durable publication, before assignment delivery
}

type Result struct {
	PID      int
	Ready    bool
	Progress workerjob.Event
	Terminal workerjob.Event
	Output   artifact.Ref
}

// Run never restarts a worker. Success requires both an authenticated successful
// terminal event and exit status zero. Every return after spawn reaps the child
// and kills any remaining process-group descendants. Log/ring files are retained
// in the caller-owned attempt directory; capabilities/sockets are removed.
func (s *Supervisor) Run(parent context.Context, p LaunchPlan) (result Result, err error) {
	if p.StartupTimeout <= 0 || p.StartupTimeout > 10*time.Minute || p.ShutdownGrace <= 0 || p.ShutdownGrace > 30*time.Second {
		return result, fmt.Errorf("invalid worker hosting limits")
	}
	if err := p.Limits.Validate(p.Assignment.RuntimeSeconds); err != nil {
		return result, err
	}
	// Freeze all slices/raw JSON before the child starts.
	binding, err := p.Assignment.Binding()
	if err != nil {
		return result, err
	}
	blob, err := json.Marshal(p.Assignment)
	if err != nil {
		return result, err
	}
	p.Assignment, err = workerjob.Decode(blob, binding)
	if err != nil || binding.BuildID != s.build {
		return result, fmt.Errorf("launch build does not match approved executable")
	}
	if p.Directory.Kind() != statehome.Worker {
		return result, fmt.Errorf("launch requires worker-owned state directory")
	}
	if err = p.Directory.Validate(); err != nil {
		return result, err
	}
	if err = checkAttemptDisk(parent, p.Directory.Dir(), p.Limits.DiskBytes); err != nil {
		return result, err
	}
	if current, e := workerjob.FileDigest(s.binary); e != nil || current != s.build {
		return result, fmt.Errorf("approved worker executable changed")
	}
	ring, err := json.Marshal(p.Assignment.RingAddresses)
	if err != nil {
		return result, err
	}
	// Exclusive files ensure an attempt directory cannot accidentally be reused.
	if _, err = os.Lstat(filepath.Join(p.Directory.Dir(), "ring.json")); !os.IsNotExist(err) {
		return result, fmt.Errorf("worker attempt directory already contains ring.json")
	}
	if err = p.Directory.WriteFile("ring.json", ring); err != nil {
		return result, err
	}
	log, err := os.OpenFile(filepath.Join(p.Directory.Dir(), "worker.log"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return result, err
	}
	defer func() { _ = log.Close() }()
	server, credential, err := local.Listen(p.Directory, local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: workerjob.ControlByteBudget, Messages: workerjob.ControlMessageBudget})
	if err != nil {
		return result, err
	}
	defer func() { _ = server.Close() }()
	ctx, cancel := context.WithTimeout(parent, time.Duration(p.Assignment.RuntimeSeconds)*time.Second)
	defer cancel()
	logs := &boundedLog{file: log, remaining: p.Limits.LogBytes, cancel: cancel}
	cmd := exec.Command(s.binary, "-mode", "managed-worker", "-worker-control-socket", server.Path(), "-worker-session-fd", "3")
	cmd.Dir = p.Directory.Dir()
	cmd.ExtraFiles = []*os.File{credential}
	cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + p.Directory.Dir(), "TMPDIR=" + p.Directory.Dir(),
		"MLX_HOSTFILE=" + filepath.Join(p.Directory.Dir(), "ring.json"), "MLX_RANK=" + strconv.Itoa(p.Assignment.View.LocalRank)}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Stdout, cmd.Stderr = logs, logs
	cmd.WaitDelay = p.ShutdownGrace
	if err = cmd.Start(); err != nil {
		return result, err
	}
	result.PID = cmd.Process.Pid
	process := ownProcess(cmd)
	wait := process.wait
	reaped := false
	var stopResources func() error
	defer func() {
		if stopResources != nil {
			err = errors.Join(err, stopResources())
		}
	}()
	defer func() {
		// Physical termination must precede the sampler join. In particular a
		// stalled filesystem read must not prevent signaling the owned group.
		cancel()
		_ = server.Close()
		if !reaped {
			_ = process.signal(syscall.SIGTERM)
			var cleanupErr error
			select {
			case cleanupErr = <-wait:
			case <-time.After(p.ShutdownGrace):
				_ = process.signal(syscall.SIGKILL)
				cleanupErr = <-wait
			}
			if errors.Is(cleanupErr, ErrReconciliationRequired) {
				err = errors.Join(err, cleanupErr)
			}
		}
	}()
	stopResources = watchResources(ctx, cancel, process, p.Directory.Dir(), p.Limits, s.sample)
	// A short job may exit between samples. Always check its retained output
	// before reporting success, including output written during shutdown.
	defer func() {
		if err == nil {
			err = checkAttemptDisk(parent, p.Directory.Dir(), p.Limits.DiskBytes)
		}
	}()
	if p.Started != nil {
		if err = p.Started(result.PID); err != nil {
			return result, fmt.Errorf("publish worker start: %w", err)
		}
	}
	startCtx, stopStart := context.WithTimeout(ctx, p.StartupTimeout)
	defer stopStart()
	type admission struct {
		conn *local.Conn
		err  error
	}
	accepted := make(chan admission, 1)
	go func() {
		c, e := server.Accept(startCtx, binding, wc.PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Getuid())})
		accepted <- admission{c, e}
	}()
	var admitted admission
	select {
	case admitted = <-accepted:
	case exitErr := <-wait:
		reaped = true
		_ = server.Close()
		<-accepted
		return result, fmt.Errorf("worker exited before admission: %w", exitErr)
	case <-ctx.Done():
		_ = server.Close()
		<-accepted
		return result, ctx.Err()
	}
	conn, err := admitted.conn, admitted.err
	if err != nil {
		return result, fmt.Errorf("worker authentication: %w", err)
	}
	select {
	case exitErr := <-wait:
		reaped = true
		return result, fmt.Errorf("worker exited during admission: %w", exitErr)
	default:
	}
	assignment, err := workerjob.Envelope(binding, 1, wc.KindAssignment, p.Assignment)
	if err != nil {
		return result, err
	}
	if err = conn.Send(startCtx, assignment); err != nil {
		return result, err
	}
	// One reader owns protocol state. Its final value is transferred by channel,
	// so the owner can cancel/reap while it is blocked on an unresponsive child.
	type outcome struct {
		value Result
		err   error
	}
	reports := make(chan outcome, 1)
	go func() {
		r, e := monitor(ctx, startCtx, conn, result, p)
		reports <- outcome{r, e}
	}()
	var report outcome
	select {
	case report = <-reports:
	case <-ctx.Done():
		_ = conn.Close()
		report = <-reports
		report.err = ctx.Err()
	case exitErr := <-wait:
		reaped = true
		// Drain a terminal event already buffered before normal process exit.
		select {
		case report = <-reports:
		case <-time.After(p.ShutdownGrace):
			_ = conn.Close()
			report = <-reports
			report.err = fmt.Errorf("worker exited without a terminal outcome")
		}
		if exitErr != nil {
			report.err = fmt.Errorf("worker exit: %w", exitErr)
		}
	}
	result = report.value
	if result.PID == 0 {
		result.PID = cmd.Process.Pid
	}
	if logs.exceeded() {
		return result, fmt.Errorf("worker log limit exceeded")
	}
	if report.err != nil {
		return result, report.err
	}
	if !reaped {
		select {
		case err = <-wait:
			reaped = true
		case <-ctx.Done():
			err = ctx.Err()
		case <-time.After(p.ShutdownGrace):
			err = fmt.Errorf("worker failed to exit after terminal outcome")
		}
	}
	return result, err
}

func monitor(ctx, start context.Context, conn *local.Conn, r Result, plan LaunchPlan) (Result, error) {
	for {
		readCtx := ctx
		if !r.Ready {
			readCtx = start
		}
		e, err := conn.Receive(readCtx)
		if err != nil {
			return r, err
		}
		if err = workerjob.CheckPayload(e, e.Kind); err != nil {
			return r, err
		}
		if e.Kind == wc.KindArtifactWrite {
			if !r.Ready || r.Output.Bytes != 0 || plan.Assignment.View.LocalRank != 0 {
				return r, fmt.Errorf("unapproved or duplicate output artifact")
			}
			r.Output, err = receiveOutputArtifact(ctx, plan.Directory, plan.Assignment.OutputMaxBytes, e, func() (wc.Envelope, error) { return conn.Receive(ctx) })
			if err != nil {
				return r, err
			}
			continue
		}
		var event workerjob.Event
		if err = workerjob.DecodePayload(e.Payload, &event); err != nil {
			return r, err
		}
		switch e.Kind {
		case wc.KindHeartbeat:
		case wc.KindReadiness:
			if r.Ready {
				return r, fmt.Errorf("duplicate worker readiness")
			}
			r.Ready = true
		case wc.KindProgress:
			if !r.Ready || event.Step <= r.Progress.Step || event.Committed < r.Progress.Committed || event.Committed > uint64(event.Step) {
				return r, fmt.Errorf("invalid worker progress")
			}
			r.Progress = event
		case wc.KindTerminalOutcome:
			r.Terminal = event
			if event.Error != "" {
				return r, errors.New(event.Error)
			}
			if !r.Ready || r.Progress.Step == 0 {
				return r, fmt.Errorf("worker completed before readiness/progress")
			}
			if plan.Assignment.OutputMaxBytes > 0 && plan.Assignment.View.LocalRank == 0 && r.Output.Bytes == 0 {
				return r, fmt.Errorf("worker omitted required final weights")
			}
			return r, nil
		default:
			return r, fmt.Errorf("unexpected worker event %s", e.Kind)
		}
	}
}

type boundedLog struct {
	mu        sync.Mutex
	file      *os.File
	remaining int64
	over      bool
	cancel    context.CancelFunc
}

func (w *boundedLog) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	n := len(p)
	if int64(n) > w.remaining {
		p = p[:w.remaining]
		w.over = true
		w.cancel()
	}
	written, err := w.file.Write(p)
	w.remaining -= int64(written)
	if err != nil {
		w.cancel()
		return written, err
	}
	// Drain/discard overflow until the owner kills the process; never allocate
	// an unbounded output buffer or block Wait on a child-held stdout pipe.
	return n, nil
}

func (w *boundedLog) exceeded() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.over
}
