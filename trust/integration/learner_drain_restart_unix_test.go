//go:build darwin || linux

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	coordinatorlocal "github.com/mrothroc/mixlab/coordinator/local"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/transport/managedtls"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
)

type drainRequestKey struct{}
type drainGate struct {
	entered chan struct{}
	used    atomic.Bool
}

func (g *drainGate) wait(ctx context.Context) error {
	if ctx.Value(drainRequestKey{}) == true && g.used.CompareAndSwap(false, true) {
		close(g.entered)
		<-ctx.Done()
		return ctx.Err()
	}
	return nil
}

type drainAllocations struct {
	coordinator.Allocations
	gate *drainGate
}

func (a drainAllocations) Ready(ctx context.Context, slot run.Slot) error {
	if err := a.Allocations.Ready(ctx, slot); err != nil {
		return err
	}
	return a.gate.wait(ctx)
}

type drainRepository struct {
	*coordinatorlocal.Repository
	gate *drainGate
}

func (r drainRepository) Save(ctx context.Context, id string, revision uint64, c coordinator.Change) error {
	if err := r.Repository.Save(ctx, id, revision, c); err != nil {
		return err
	}
	if c.Receipt != nil {
		return r.gate.wait(ctx)
	}
	return nil
}

type drainHTTP struct {
	semantic, artifacts http.Handler
	worker, operation   string
	gate                bool
	polls               atomic.Int32
	mu                  sync.Mutex
	requests            [][]byte
	statuses            []int
	unavailable         chan struct{}
	observe             func(learnerapi.Request, int, string)
}

func (h *drainHTTP) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != learnerapi.Route {
		h.artifacts.ServeHTTP(w, r)
		return
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, learnerapi.MaxMessage+1))
	if err != nil {
		http.Error(w, "read", 400)
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(b))
	var q learnerapi.Request
	if err := json.Unmarshal(b, &q); err != nil {
		http.Error(w, "decode", 400)
		return
	}
	target := q.Call.WorkerID == h.worker && q.Call.Operation == h.operation
	if target && h.operation == "poll" {
		target = h.polls.Add(1) >= 2
	}
	if target && h.gate {
		r = r.WithContext(context.WithValue(r.Context(), drainRequestKey{}, true))
	}
	recorder := httptest.NewRecorder()
	h.semantic.ServeHTTP(capturedLearnerResponse{recorder, w}, r)
	if h.observe != nil {
		h.observe(q, recorder.Code, recorder.Body.String())
	}
	if target {
		h.mu.Lock()
		h.requests = append(h.requests, bytes.Clone(b))
		h.statuses = append(h.statuses, recorder.Code)
		h.mu.Unlock()
	}
	for k, values := range recorder.Header() {
		w.Header()[k] = values
	}
	w.WriteHeader(recorder.Code)
	_, _ = w.Write(recorder.Body.Bytes())
	if target && recorder.Code == http.StatusServiceUnavailable {
		w.(http.Flusher).Flush()
		select {
		case h.unavailable <- struct{}{}:
		default:
		}
	}
}

func drainHTTPServer(t *testing.T, p *managedtls.Policy, h http.Handler, base context.Context, address string) *httptest.Server {
	t.Helper()
	s := httptest.NewUnstartedServer(h)
	if address != "" {
		check(t, s.Listener.Close())
		listener, err := net.Listen("tcp", address)
		check(t, err)
		s.Listener = listener
	}
	s.Config.BaseContext = func(net.Listener) context.Context { return base }
	s.Config.ErrorLog = log.New(io.Discard, "", 0)
	s.TLS = p.ServerConfig()
	s.StartTLS()
	t.Cleanup(s.Close)
	return s
}

func TestManagedLearnerIPCAuthenticatedDrainRestart(t *testing.T) {
	for _, operation := range []string{"poll", "submit"} {
		t.Run(operation, func(t *testing.T) { testManagedLearnerDrain(t, operation) })
	}
}

func testManagedLearnerDrain(t *testing.T, operation string) {
	exe, err := os.Executable()
	check(t, err)
	build, err := workerjob.FileDigest(exe)
	check(t, err)
	supervisor, err := workerhost.New(exe, build)
	check(t, err)
	x, repoPath, config := drainRelayFixture(t, build)
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	serviceCtx, stopService := context.WithCancel(context.Background())
	defer stopService()
	gate := &drainGate{entered: make(chan struct{})}
	if operation == "poll" {
		x.model.Allocations = drainAllocations{x.model.Allocations, gate}
	} else {
		x.model.Repository = drainRepository{x.model.Repository.(*coordinatorlocal.Repository), gate}
	}
	artifacts, err := x.g.Handler()
	check(t, err)
	observe := func(q learnerapi.Request, code int, body string) {
		if code != 200 && q.Call.Operation != "receipt" {
			t.Logf("HTTP worker=%s operation=%s status=%d body=%s", q.Call.WorkerID, q.Call.Operation, code, body)
		}
	}
	old := &drainHTTP{semantic: retryRelayHandler(t, x), artifacts: artifacts, worker: x.bindings[0].Worker, operation: operation, gate: true, unavailable: make(chan struct{}, 32), observe: observe}
	server := drainHTTPServer(t, x.g.Policy, old, serviceCtx, "")
	address, endpoint := server.Listener.Addr().String(), server.URL
	var policies atomic.Int32
	clients := make([]*clusterapp.LearnerClient, 2)
	for i := range clients {
		o := x.options[i]
		o.Endpoint = endpoint
		policy := o.Policy
		o.Policy = func() (*managedtls.Policy, error) { policies.Add(1); return policy() }
		clients[i], err = clusterapp.NewLearnerClientWithPolicy(o)
		check(t, err)
	}
	type result struct {
		index  int
		result workerhost.Result
		err    error
	}
	done := make(chan result, 2)
	acknowledged := make(chan struct{}, 2)
	var wg sync.WaitGroup
	defer func() { cancel(); wg.Wait() }()
	paths := make([]string, 2)
	logs := make([]string, 2)
	var pids [2]atomic.Int64
	launch := func(i int) {
		directory, privatePath := drainWorkerDirectory(t), drainWorkerDirectory(t)
		paths[i], logs[i] = privatePath.Dir(), filepath.Join(directory.Dir(), "worker.log")
		b := x.bindings[i]
		a := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: b.Job, AttemptID: b.Attempt, RunID: b.Run, PlanHash: b.PlanHash,
			Member: x.s.Plan.Slots[i].Member, Contract: x.s.Plan.Contract, Coordinator: x.s.Plan.Coordinator, Config: config,
			DatasetSelector: "drain-test", TrainPattern: "/unused/synthetic.bin", RuntimeSeconds: 40, PrivateMaxBytes: 1 << 20, Cohort: &x.s.Plan.Cohort}
		ipc := &learneripc.Server{Remote: clients[i], MaxBytes: 1 << 20}
		var submitted, reported bool
		relay := func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
			var q struct {
				Operation string
				Call      *private.RelayRequest
			}
			if err := json.Unmarshal(e.Payload, &q); err != nil {
				return wc.Envelope{}, err
			}
			// Advance only after both children have received their receipts and
			// re-entered Poll, not while a submit response is still in flight.
			if q.Operation == "poll" && submitted && !reported {
				reported = true
				acknowledged <- struct{}{}
			}
			out, err := ipc.Handle(ctx, e)
			if err == nil && q.Call != nil && q.Call.Operation == "submit" {
				var reply struct{ Reply *private.RelayResponse }
				if json.Unmarshal(out.Payload, &reply) == nil && reply.Reply != nil && reply.Reply.Receipt != nil {
					submitted = true
				}
			}
			return out, err
		}
		p := workerhost.LaunchPlan{Learner: &a, Directory: directory, LearnerPrivateDirectory: privatePath,
			StartupTimeout: 5 * time.Second, ShutdownGrace: 2 * time.Second, Limits: contract.Limits{CPUSeconds: 60, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20},
			Relay: relay, Started: func(pid int) error { pids[i].Store(int64(pid)); return nil }}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer ipc.Close()
			r, err := supervisor.Run(ctx, p)
			done <- result{i, r, err}
		}()
	}
	launch(0)
	failResult := func(r result) {
		b, _ := os.ReadFile(logs[r.index])
		t.Fatalf("worker %d exited before recovery: %+v %v\n%s", r.index, r.result, r.err, b)
	}
	await := func(ch <-chan struct{}) {
		select {
		case <-ch:
		case r := <-done:
			failResult(r)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	await(gate.entered)
	pid := int(pids[0].Load())
	if pid == 0 {
		t.Fatal("drain preceded real child startup")
	}
	check(t, syscall.Kill(pid, 0))
	before, err := x.model.Repository.Load(ctx, x.s.Plan.Cohort.RunID)
	check(t, err)
	if operation == "submit" {
		found := false
		for _, u := range before.Updates {
			if u.WorkerID == x.bindings[0].Worker {
				found = true
			}
		}
		if !found {
			t.Fatal("submit drain preceded durable acceptance")
		}
	}
	// Cancel the service only after an authenticated workload is inside the
	// gateway. Observe the actual 503 before replacing the listener; a canceled
	// BaseContext may reject subsequent TLS handshakes before reaching HTTP.
	stopService()
	await(old.unavailable)
	check(t, syscall.Kill(pid, 0))
	select {
	case r := <-done:
		failResult(r)
	default:
	}
	drainCtx, stopDrain := context.WithTimeout(ctx, 5*time.Second)
	err = server.Config.Shutdown(drainCtx)
	stopDrain()
	check(t, err)
	server.Close()
	repo, err := coordinatorlocal.Open(repoPath)
	check(t, err)
	x.model.Repository, x.g.Repository = repo, repo
	// Keep fixture allocations live through child exit. Eager release based on
	// a second state read would race an already-authorized terminal poll.
	x.model.Allocations = relayAllocations{}
	artifacts, err = x.g.Handler()
	check(t, err)
	fresh := &drainHTTP{semantic: retryRelayHandler(t, x), artifacts: artifacts, worker: x.bindings[0].Worker, operation: operation, unavailable: make(chan struct{}, 32), observe: observe}
	// The first poll on the fresh handler is already a retried live poll.
	fresh.polls.Store(1)
	restarted := drainHTTPServer(t, x.g.Policy, fresh, context.Background(), address)
	if restarted.URL != endpoint {
		t.Fatal("restart changed coordinator origin")
	}
	await(acknowledged)
	// This regression targets one live semantic request, not concurrent
	// coordinator CAS writers. Start the second worker after the first receipt.
	launch(1)
	await(acknowledged)
	for range 3 {
		check(t, x.model.Advance(ctx, x.s.Plan.Cohort.RunID))
	}
	for range 2 {
		select {
		case r := <-done:
			if r.err != nil || !r.result.Ready || r.result.PID != int(pids[r.index].Load()) || r.result.Terminal.Committed != 1 {
				failResult(r)
			}
			if syscall.Kill(r.result.PID, 0) != syscall.ESRCH {
				t.Fatal("managed child was not reaped")
			}
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	final, err := repo.Load(ctx, x.s.Plan.Cohort.RunID)
	check(t, err)
	if final.Phase != coordinator.Complete || final.Round != 1 || len(final.Updates) != 2 || len(final.Starts) != 2 || len(final.Attempts) != 0 || final.Revision != x.s.Revision+8 {
		t.Fatalf("round was duplicated, replaced, or incomplete: phase=%s round=%d updates=%d starts=%d attempts=%d revision=%d", final.Phase, final.Round, len(final.Updates), len(final.Starts), len(final.Attempts), final.Revision)
	}
	for i, path := range paths {
		var j struct {
			Completed private.PrivateState
			Receipt   private.Receipt
		}
		b, err := os.ReadFile(filepath.Join(path, "head.json"))
		check(t, err)
		check(t, json.Unmarshal(b, &j))
		if j.Completed.Assignment != x.s.Assignments[i] || j.Completed.Replay.AttemptStep != x.s.Assignments[i].InnerSteps {
			t.Fatal("private round or sampler advanced twice")
		}
		r, err := repo.Receipt(ctx, x.s.Plan.Cohort.RunID, j.Receipt.UpdateID)
		check(t, err)
		if r != j.Receipt {
			t.Fatal("learner receipt differs from durable coordinator receipt")
		}
	}
	old.mu.Lock()
	defer old.mu.Unlock()
	fresh.mu.Lock()
	defer fresh.mu.Unlock()
	if len(old.statuses) < 1 || old.statuses[0] != 503 || len(fresh.statuses) == 0 || fresh.statuses[0] != 200 || policies.Load() < 4 {
		t.Fatal("drain/retry/restart path not exercised", old.statuses, fresh.statuses)
	}
	for _, b := range append(old.requests, fresh.requests...) {
		if !bytes.Equal(b, old.requests[0]) {
			t.Fatal("semantic retry changed request bytes")
		}
	}
	t.Logf("same PID %d survived authenticated %s 503 drain; one update per worker; revision %d", pid, operation, final.Revision)
}
