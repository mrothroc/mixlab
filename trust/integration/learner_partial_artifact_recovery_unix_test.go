//go:build darwin || linux

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestManagedLearnerPartialArtifactRecovery(t *testing.T) {
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
	grantsPath, ledgerPath := artifactPath(t), artifactPath(t)
	wire := &partialArtifactWire{cut: make(chan int, 1)}
	policy, handler := partialArtifactHandler(t, f, grantsPath, ledgerPath)
	server := serveCapabilities(t, policy, wire.wrap(t, handler))
	endpoint, address := server.URL, server.Listener.Addr().String()
	x, err := f.x.store.ActiveExecution(ctx)
	check(t, err)
	initialAttempt := x.Approval.AttemptID()
	checkpoint := primePartialArtifactCheckpoint(t, f, privatePath, partialArtifactClient(t, f, *x, endpoint))
	before := f.state(t)
	if len(before.Starts) != 1 || len(before.Updates) != 0 || wire.reads.Load() != 1 {
		t.Fatal("fixture lacks acknowledged private start", before.Starts, before.Updates)
	}
	wire.armed.Store(true)
	readErrors := make(chan error, 2)
	type diagnostic struct {
		operation string
		err       error
	}
	var diagnosticsMu sync.Mutex
	var diagnostics []diagnostic
	type outcome struct {
		value contract.Outcome
		err   error
	}
	launch := func() (<-chan int, <-chan outcome, string) {
		x, e := f.x.store.ActiveExecution(ctx)
		check(t, e)
		directory := drainWorkerDirectory(t)
		host, e := workerhost.NewAttemptStore(directory)
		check(t, e)
		local, e := workerhost.NewLocalRunner(supervisor, directory, 5*time.Second, 2*time.Second)
		check(t, e)
		peer := &partialArtifactPeer{partialArtifactClient(t, f, *x, endpoint), f, readErrors}
		ipc := &learneripc.Server{Remote: peer, MaxBytes: 1 << 20, OnError: func(operation string, err error) {
			diagnosticsMu.Lock()
			defer diagnosticsMu.Unlock()
			diagnostics = append(diagnostics, diagnostic{operation, err})
		}}
		local.LearnerRelay, local.LearnerPrivateDirectory = ipc.Handle, privatePath
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
	awaitExit := func(done <-chan outcome, path string, kind string) contract.Outcome {
		select {
		case r := <-done:
			check(t, r.err)
			if r.value.Kind != kind || !r.value.NoChild {
				b, _ := os.ReadFile(path)
				t.Fatalf("unexpected child outcome: %+v log=%s", r.value, b)
			}
			return r.value
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
		return contract.Outcome{}
	}
	pids, done, workerLog := launch()
	firstPID := awaitPID(pids, done)
	failed := awaitExit(done, workerLog, contract.Failed)
	if !strings.Contains(failed.Error, "operation rejected") {
		t.Fatal("failure did not traverse sanitized learner IPC", failed.Error)
	}
	if err := syscall.Kill(firstPID, 0); !errors.Is(err, syscall.ESRCH) {
		t.Fatal("failed learner was not reaped", err)
	}
	select {
	case n := <-wire.cut:
		if n <= 0 || uint64(n) >= f.assignment.BaseModel.Ref.Bytes {
			t.Fatal("response was not partially delivered", n)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	var rawReadError error
	select {
	case rawReadError = <-readErrors:
		if !errors.Is(rawReadError, io.ErrUnexpectedEOF) || errors.Is(rawReadError, private.ErrUnavailable) {
			t.Fatal("wrong raw stream failure classification", rawReadError)
		}
	default:
		t.Fatal("missing agent-side raw stream error")
	}
	diagnosticsMu.Lock()
	logged := append([]diagnostic(nil), diagnostics...)
	diagnosticsMu.Unlock()
	readDiagnostics := 0
	for _, d := range logged {
		if d.operation == "read_begin" {
			readDiagnostics++
			if d.err != rawReadError || !errors.Is(d.err, io.ErrUnexpectedEOF) || errors.Is(d.err, private.ErrUnavailable) {
				t.Fatal("local IPC diagnostic changed raw stream error", d.err)
			}
		}
	}
	if readDiagnostics != 1 {
		t.Fatal("partial artifact failure was not diagnosed exactly once locally", readDiagnostics)
	}
	childLog, err := os.ReadFile(workerLog)
	check(t, err)
	if strings.Contains(string(childLog), io.ErrUnexpectedEOF.Error()) || strings.Contains(failed.Error, io.ErrUnexpectedEOF.Error()) {
		t.Fatal("local artifact diagnostic leaked into child log or public outcome")
	}
	if wire.reads.Load() != 2 {
		t.Fatal("caller-owned artifact stream was replayed", wire.reads.Load())
	}
	head, err := privatePath.ReadFileLimit("head.json", 4<<20)
	check(t, err)
	if !bytes.Equal(head, checkpoint) || !reflect.DeepEqual(f.state(t), before) {
		t.Fatal("failed download changed private history or coordinator work")
	}

	// The listener and gateway composition are replaced, with durable grants,
	// ledger and run state retained. No installed service or remote host is used.
	server.Close()
	for range 4 {
		_, err = f.recovery(t).Tick(ctx, f.assignment.Binding.RunID)
		check(t, err)
	}
	s := f.state(t)
	if s.Recovery != nil || len(s.Attempts) != 1 || s.Attempts[0].PreviousAttemptID != initialAttempt || s.Attempts[0].PrivateHash != before.Starts[0].PrivateHash || !reflect.DeepEqual(s.Plan.Slots, before.Plan.Slots) || !reflect.DeepEqual(s.Assignments, before.Assignments) {
		t.Fatal("recovery changed allocation/round/private checkpoint", s.Recovery, s.Attempts)
	}
	x, err = f.x.store.ActiveExecution(ctx)
	check(t, err)
	if x.Learner.Assignment.ExpectedPrivateHash != before.Starts[0].PrivateHash || x.Workload.Generation != 2 || x.Approval.AttemptID() == initialAttempt || f.x.initializations.Load() != 2 {
		t.Fatal("replacement was not bound to original private state")
	}
	f.model.Repository, err = coordinatorlocal.Open(f.repoPath)
	check(t, err)
	policy, handler = partialArtifactHandler(t, f, grantsPath, ledgerPath)
	listener, err := net.Listen("tcp", address)
	check(t, err)
	restarted := httptest.NewUnstartedServer(wire.wrap(t, handler))
	check(t, restarted.Listener.Close())
	restarted.Listener = listener
	restarted.TLS = policy.ServerConfig()
	restarted.Config.ErrorLog = log.New(io.Discard, "", 0)
	restarted.StartTLS()
	defer restarted.Close()
	pids, done, workerLog = launch()
	secondPID := awaitPID(pids, done)
	if secondPID == firstPID {
		t.Fatal("replacement reused failed process")
	}
	awaitExit(done, workerLog, contract.Exited)
	s = f.state(t)
	if s.Phase != coordinator.Complete || len(s.Starts) != 1 || len(s.Updates) != 1 || len(s.Attempts) != 1 || s.Recovery != nil {
		t.Fatal("recovered round did not complete exactly once", s.Phase, s.Starts, s.Updates, s.Attempts)
	}
	// A distinct seed distinguishes genuine private restore from a fresh
	// engine that happens to produce one update. Both runtime re-executions in
	// the child must return the original receipt without another Train call.
	head, err = privatePath.ReadFileLimit("head.json", 4<<20)
	check(t, err)
	var journal struct {
		Start     private.PrivateState `json:"start"`
		Completed private.PrivateState `json:"completed"`
		Receipt   private.Receipt      `json:"receipt"`
	}
	check(t, json.Unmarshal(head, &journal))
	if journal.Start.Replay.Seed != 99 || journal.Completed.Replay.Seed != 99 || journal.Start.Replay.AttemptStep != 0 || journal.Completed.Replay.AttemptStep != f.assignment.InnerSteps {
		t.Fatal("private replay state was reset or advanced twice", journal.Start.Replay, journal.Completed.Replay)
	}
	receipt, err := f.model.Repository.Receipt(ctx, f.assignment.Binding.RunID, s.Updates[0].ID)
	check(t, err)
	if receipt != journal.Receipt || s.Updates[0].PrivateHash != receipt.PrivateHash {
		t.Fatal("durable receipt differs from recovered private state")
	}
}

func TestLearnerPartialArtifactBodyIsNotReplayed(t *testing.T) {
	// Exceed artifact.Copy's buffer so some authenticated response bytes reach
	// the caller-owned destination before the truncated next chunk fails.
	x := newLearnerRelayModel(t, []optimizer.Tensor{{Name: "weight", Shape: []int{40000}, Values: make([]float32, 40000)}}, precision.Policy{ComputeDType: "float32", Backend: "metal"})
	handler, err := x.g.Handler()
	check(t, err)
	semantic := retryRelayHandler(t, x)
	mux := http.NewServeMux()
	mux.Handle(learnerapi.Route, semantic)
	mux.Handle("/v1/diloco/artifacts/", handler)
	wire := &partialArtifactWire{cut: make(chan int, 1)}
	wire.armed.Store(true)
	server := serveCapabilities(t, x.g.Policy, wire.wrap(t, mux))
	c := retryRelayClient(t, x, server.URL, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var body bytes.Buffer
	a := x.s.Assignments[0]
	err = c.Read(ctx, a, a.BaseModel, &body)
	if !errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, private.ErrUnavailable) {
		t.Fatal("partial response was not classified as a terminal stream error", err)
	}
	if wire.reads.Load() != 1 || body.Len() != 64<<10 || !bytes.Equal(body.Bytes(), x.body[:body.Len()]) {
		t.Fatal("stream replayed, prefix changed, or partial bytes were not delivered", wire.reads.Load(), body.Len())
	}
	select {
	case n := <-wire.cut:
		if n <= body.Len() || n >= len(x.body) {
			t.Fatal("wrong partial wire length", n)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
