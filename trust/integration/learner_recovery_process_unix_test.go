//go:build darwin || linux

package integration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

// Only numerical work and the learner's remote transport are test adapters.
// The real guardian, supervisor, local IPC, private journal, node HTTP recovery,
// credential issuer and coordinator recovery composition are exercised here.
type recoveryProcessPeer struct {
	f         *recoveryFixture
	execution nodeagent.LocalExecution
	start     chan struct{}
	release   <-chan struct{}
}

func (p *recoveryProcessPeer) Poll(ctx context.Context) (private.RelayResponse, error) {
	m := p.f.model
	id := p.f.assignment.Binding.RunID
	if err := m.Tick(ctx, id, 200); err != nil {
		return private.RelayResponse{}, err
	}
	s, err := m.Repository.Load(ctx, id)
	out := private.RelayResponse{Format: "mixlab_learner_reply_v1"}
	if err != nil {
		return out, err
	}
	switch s.Phase {
	case coordinator.Open:
		out.Assignment = &s.Assignments[0]
	case coordinator.Evaluating:
		out.Evaluation = s.Evaluation
	case coordinator.Complete:
		out.Done = true
	case coordinator.Failed:
		return out, fmt.Errorf("run failed: %s", s.Failure)
	}
	return out, nil
}
func (p *recoveryProcessPeer) Call(ctx context.Context, q private.RelayRequest) (private.RelayResponse, error) {
	out := private.RelayResponse{Format: "mixlab_learner_reply_v1"}
	if err := q.Validate(); err != nil {
		return out, err
	}
	x := p.execution
	w := x.Workload
	allocation := x.Learner.Signed.Manifest.Slot.Allocation
	var err error
	switch q.Operation {
	case "start":
		err = p.f.model.RecordStartSubmission(ctx, private.StartSubmission{Format: "mixlab_start_submission_v1", Start: *q.Start, PrincipalID: w.Workload, AttemptID: w.Attempt, WorkloadGeneration: w.Generation, Allocation: allocation})
		if err == nil && p.start != nil {
			close(p.start)
			p.start = nil
			select {
			case <-p.release:
			case <-ctx.Done():
				return out, ctx.Err()
			}
		}
	case "submit":
		var receipt private.Receipt
		receipt, err = p.f.model.AcceptSubmission(ctx, private.Submission{Format: "mixlab_update_submission_v1", Update: *q.Update, PrincipalID: w.Workload, AttemptID: w.Attempt, WorkloadGeneration: w.Generation, Allocation: allocation})
		out.Receipt = &receipt
	case "receipt":
		var receipt private.Receipt
		receipt, err = p.f.model.Repository.Receipt(ctx, q.Binding.RunID, q.Update.ID)
		if errors.Is(err, coordinator.ErrNotFound) {
			err = private.ErrReceiptPending
		}
		if err == nil {
			out.Receipt = &receipt
		}
	case "evaluate":
		err = p.f.model.Advance(ctx, q.Binding.RunID)
	default:
		err = fmt.Errorf("unexpected process operation")
	}
	return out, err
}
func (p *recoveryProcessPeer) Read(ctx context.Context, _ run.Assignment, b run.Blob, w io.Writer) error {
	return p.f.model.Artifacts.(coordinatorlocal.Artifacts).Store.Copy(ctx, b.Ref, w)
}
func (p *recoveryProcessPeer) Write(ctx context.Context, _ run.Assignment, b run.Blob, r io.Reader) error {
	return p.f.model.Artifacts.(coordinatorlocal.Artifacts).Store.PutVerified(ctx, b.Ref, r)
}
func (p *recoveryProcessPeer) ReadEvaluation(ctx context.Context, a run.EvaluationAssignment, w io.Writer) error {
	return p.Read(ctx, run.Assignment{}, a.Model, w)
}

type recoveryStartedRunner struct {
	workerhost.Runner
	store *nodeagent.Store
	host  *workerhost.AttemptStore
	pid   chan<- int
}

func (r recoveryStartedRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	return r.Runner.Run(ctx, a, func(pid int) error {
		if err := started(pid); err != nil {
			return err
		}
		state, err := r.host.Status()
		if err != nil {
			return err
		}
		if err = r.store.ApplyLearnerOutcome(ctx, *state.Started); err != nil {
			return err
		}
		r.pid <- pid
		return nil
	})
}

func TestProductionLearnerRecoveryAfterRealProcessKill(t *testing.T) {
	exe, err := os.Executable()
	check(t, err)
	build, err := workerjob.FileDigest(exe)
	check(t, err)
	supervisor, err := workerhost.New(exe, build)
	check(t, err)
	f := newRecoveryFixtureWithBuild(t, build, false)
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	privatePath := drainWorkerDirectory(t)
	committed, release := make(chan struct{}), make(chan struct{})
	type outcome struct {
		value contract.Outcome
		err   error
	}
	launch := func(first bool) (<-chan int, <-chan outcome, string) {
		x, e := f.x.store.ActiveExecution(ctx)
		check(t, e)
		directory := drainWorkerDirectory(t)
		host, e := workerhost.NewAttemptStore(directory)
		check(t, e)
		local, e := workerhost.NewLocalRunner(supervisor, directory, 5*time.Second, 2*time.Second)
		check(t, e)
		peer := &recoveryProcessPeer{f: f, execution: *x}
		if first {
			peer.start = committed
			peer.release = release
		}
		ipc := &learneripc.Server{Remote: peer, MaxBytes: 1 << 20}
		local.LearnerRelay = ipc.Handle
		local.LearnerPrivateDirectory = privatePath
		guardian, e := workerhost.NewGuardianRunner(local, exe, build)
		check(t, e)
		pids, done := make(chan int, 1), make(chan outcome, 1)
		go func() {
			defer ipc.Close()
			out, e := host.Run(ctx, *x.Approval, recoveryStartedRunner{guardian, f.x.store, host, pids})
			if e == nil {
				e = f.x.store.ApplyLearnerOutcome(ctx, out)
			}
			done <- outcome{out, e}
		}()
		return pids, done, filepath.Join(directory.Dir(), "worker.log")
	}
	awaitPID := func(pids <-chan int, done <-chan outcome) int {
		select {
		case pid := <-pids:
			return pid
		case r := <-done:
			t.Fatalf("child never started: %+v %v", r.value, r.err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
		return 0
	}
	pids, done, log := launch(true)
	firstPID := awaitPID(pids, done)
	select {
	case <-committed:
	case r := <-done:
		b, _ := os.ReadFile(log)
		t.Fatalf("child exited before start commit: %+v %v %s", r.value, r.err, b)
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	check(t, syscall.Kill(firstPID, syscall.SIGKILL))
	close(release)
	select {
	case r := <-done:
		check(t, r.err)
		if r.value.Kind != contract.Failed || !r.value.NoChild {
			t.Fatal("kill did not produce physical terminal evidence", r.value)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if err := syscall.Kill(firstPID, 0); !errors.Is(err, syscall.ESRCH) {
		t.Fatal("old child still exists", err)
	}
	for range 4 {
		_, err = f.recovery(t).Tick(ctx, f.assignment.Binding.RunID)
		check(t, err)
	}
	s := f.state(t)
	if len(s.Attempts) != 1 || s.Recovery != nil || s.Attempts[0].PrivateHash != s.Starts[0].PrivateHash {
		t.Fatal("recovery did not bind original private start", s.Recovery, s.Attempts)
	}
	pids, done, log = launch(false)
	secondPID := awaitPID(pids, done)
	if secondPID == firstPID {
		t.Fatal("retry reused process")
	}
	select {
	case r := <-done:
		check(t, r.err)
		if r.value.Kind != contract.Exited {
			b, _ := os.ReadFile(log)
			t.Fatalf("recovered child failed: %+v %s", r.value, b)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	s = f.state(t)
	if s.Phase != coordinator.Complete || len(s.Updates) != 1 || len(s.Attempts) != 1 {
		t.Fatal("process recovery did not complete exactly once", s.Phase, len(s.Updates), len(s.Attempts))
	}
}
