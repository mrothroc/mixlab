//go:build darwin || linux

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
)

type relayEngine struct {
	state learner.Snapshot
	calls *int
}

func relayClone[T any](v T) T {
	b, _ := json.Marshal(v)
	var out T
	_ = json.Unmarshal(b, &out)
	return out
}
func (e *relayEngine) Snapshot() (learner.Snapshot, error) { return relayClone(e.state), nil }
func (e *relayEngine) Restore(s learner.Snapshot) error    { e.state = relayClone(s); return nil }
func (e *relayEngine) Install(v []optimizer.Tensor) error  { e.state.Model = relayClone(v); return nil }
func (e *relayEngine) Close()                              {}
func (e *relayEngine) Train(_ context.Context, n uint64) (learner.Counts, error) {
	*e.calls++
	for range n {
		e.state.Replay.AttemptStep++
		e.state.Model[0].Values[0] -= float32(e.state.Replay.AttemptStep) * .125
	}
	e.state.Optimizer = []byte(fmt.Sprintf("moment-%d", e.state.Replay.AttemptStep))
	e.state.Sampler = []byte(fmt.Sprintf("cursor-%d", e.state.Replay.AttemptStep))
	return learner.Counts{Attempted: n, Committed: n, Tokens: 8 * n, Examples: 2 * n, Loss: 2}, nil
}

type interruptedRelay struct {
	*clusterapp.LearnerClient
	operation string
}

func (r *interruptedRelay) Call(ctx context.Context, q private.RelayRequest) (private.RelayResponse, error) {
	v, err := r.LearnerClient.Call(ctx, q)
	if err == nil && q.Operation == r.operation {
		r.operation = ""
		return private.RelayResponse{}, io.ErrUnexpectedEOF
	}
	return v, err
}
func relayWorkerPath(t *testing.T) statehome.Path {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	check(t, err)
	return p
}

// Uses the production mutual local admission and framed dispatcher, not a Go
// channel standing in for IPC. Both learner engines run in the test process; the
// only wire credential is the single-attempt local bootstrap pipe capability.
func relayIPCObserved(t *testing.T, b trust.WorkloadBinding, remote learneripc.Remote, observe func(wc.Envelope)) (*learneripc.Client, func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	limits := local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 64 << 20, Messages: 10000}
	s, bootstrap, err := local.Listen(relayWorkerPath(t), limits)
	check(t, err)
	fd, err := syscall.Dup(int(bootstrap.Fd()))
	check(t, err)
	childPipe := os.NewFile(uintptr(fd), "local-bootstrap")
	type connected struct {
		c   *local.Conn
		err error
	}
	ready := make(chan connected, 1)
	go func() { c, e := local.Connect(ctx, s.Path(), childPipe, limits); ready <- connected{c, e} }()
	binding := wc.Binding{JobID: b.Job, AttemptID: b.Attempt, BinaryID: "mixlab", BuildID: "test-build", AssignmentDigest: [32]byte{1}}
	agentConn, err := s.Accept(ctx, binding, wc.PeerIdentity{PID: os.Getpid(), UID: uint32(os.Geteuid())})
	check(t, err)
	workerConn := <-ready
	check(t, workerConn.err)
	agent, err := local.NewDispatcher(ctx, agentConn, local.DispatcherOptions{ReadTimeout: 2 * time.Minute})
	check(t, err)
	worker, err := local.NewDispatcher(ctx, workerConn.c, local.DispatcherOptions{ReadTimeout: 2 * time.Minute})
	check(t, err)
	server := &learneripc.Server{Remote: remote, MaxBytes: 1 << 20}
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer server.Close()
		for e := range agent.Events() {
			if observe != nil {
				observe(e)
			}
			response, err := server.Handle(ctx, e)
			if err != nil {
				_ = agent.Close()
				return
			}
			if err = agent.Reply(ctx, e, response); err != nil {
				return
			}
			if observe != nil {
				observe(response)
			}
		}
	}()
	closeAll := func() { cancel(); _ = worker.Close(); _ = agent.Close(); _ = s.Close(); <-done }
	t.Cleanup(closeAll)
	return &learneripc.Client{Session: worker}, closeAll
}

func (x *relayFixture) replaceRuntimeAttempt(t *testing.T, index int, expected string) {
	t.Helper()
	b := x.bindings[index]
	previous := b.Attempt
	b.Attempt = id(t)
	b.Generation++
	b.Principal = id(t)
	reg := x.s.Commit.Registrations[index]
	check(t, x.model.BindAttempt(context.Background(), run.AttemptBinding{Format: "mixlab_learner_attempt_v1", Binding: x.s.Commit.Binding, WorkerID: b.Worker, PreviousAttemptID: previous, AttemptID: b.Attempt, WorkloadGeneration: b.Generation, Allocation: reg.Slot.Allocation, NodeAuthorizationHash: strings.Repeat("b", 64), PrivateHash: expected, PrecisionHash: reg.PrecisionHash}, relayRecovery{}))
	p, err := managedtls.New(x.issue(t, b), x.f.verify(trust.Coordinator, x.f.serverID), x.f.clock)
	check(t, err)
	o := x.options[index]
	o.Workload = b
	o.Policy = func() (*managedtls.Policy, error) { return p, nil }
	c, err := clusterapp.NewLearnerClientWithPolicy(o)
	check(t, err)
	x.clients[index] = c
	x.bindings[index] = b
}

func TestLearnerRelayRuntimeIPCRecovery(t *testing.T) {
	x := newLearnerRelay(t)
	runLearnerRelayRecovery(t, x, relayRuntimeScenario{
		factory: func(i int, calls *int) learner.Factory {
			a := x.s.Assignments[i]
			return func(model []optimizer.Tensor) (learner.Engine, error) {
				return &relayEngine{calls: calls, state: learner.Snapshot{Model: relayClone(model), Optimizer: []byte("moment-0"), Sampler: []byte("cursor-0"), Schedule: []byte("schedule"), Metadata: []byte("metadata"), Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: a.Binding.RunID, WorkerID: a.WorkerID, PartitionID: a.PartitionID, Seed: 7}, Precision: precision.Policy{ComputeDType: "float32", Backend: "metal"}}}, nil
			}
		},
		evaluate: func(_ context.Context, _ run.EvaluationAssignment, model []optimizer.Tensor) (float64, uint64, error) {
			if len(model) != 1 {
				t.Fatal("wrong evaluation model")
			}
			return 1, 8, nil
		},
		verify: func(_ int, path statehome.Path, p private.PrivateState) {
			for _, part := range []struct{ hash, want string }{{p.Optimizer.SHA256, "moment-4"}, {p.Sampler.SHA256, "cursor-4"}} {
				b, err := path.ReadFileLimit(part.hash, 1024)
				check(t, err)
				if string(b) != part.want {
					t.Fatal("private state reset")
				}
			}
		},
	})
}

type relayRuntimeScenario struct {
	factory  func(int, *int) learner.Factory
	evaluate learner.EvaluateModel
	verify   func(int, statehome.Path, private.PrivateState)
	observe  func(wc.Envelope)
}

func runLearnerRelayRecovery(t *testing.T, x *relayFixture, scenario relayRuntimeScenario) {
	t.Helper()
	ctx := context.Background()
	runtimes := make([]learner.Runtime, 2)
	paths := make([]statehome.Path, 2)
	locals := make([]*learneripc.Client, 2)
	closes := make([]func(), 2)
	calls := []int{0, 0}
	for i := range runtimes {
		paths[i] = relayWorkerPath(t)
		store, err := learner.Open(paths[i], 1<<20)
		check(t, err)
		operation := "start"
		if i == 1 {
			operation = "submit"
		}
		locals[i], closes[i] = relayIPCObserved(t, x.bindings[i], &interruptedRelay{LearnerClient: x.clients[i], operation: operation}, scenario.observe)
		a := x.s.Assignments[i]
		member := x.s.Plan.Slots[i].Member
		runtimes[i] = learner.Runtime{Store: store, Peer: locals[i], Now: func() int64 { return x.f.clock().Unix() }, Scope: learner.Scope{RunID: a.Binding.RunID, PlanHash: a.Binding.PlanHash, CohortHash: a.Binding.CohortHash, Generation: a.Binding.Generation, WorkerID: a.WorkerID, PartitionID: a.PartitionID, ParticipantID: member.ParticipantID, IslandID: member.IslandID, Contract: a.Contract, Coordinator: a.Coordinator}}
		runtimes[i].Factory = scenario.factory(i, &calls[i])
	}
	for round := range 2 {
		for i := range runtimes {
			r, err := locals[i].Poll(ctx)
			check(t, err)
			if r.Assignment == nil {
				t.Fatal("missing local assignment")
			}
			a := *r.Assignment
			if round == 0 {
				if _, err = runtimes[i].Execute(ctx, a); err == nil {
					t.Fatal("response loss not injected")
				}
				if calls[i] != i {
					t.Fatalf("unexpected training before interruption: %d", calls[i])
				}
				closes[i]()
				s, err := x.model.Repository.Load(ctx, a.Binding.RunID)
				check(t, err)
				expected := ""
				for _, v := range s.Starts {
					if v.WorkerID == a.WorkerID {
						expected = v.PrivateHash
					}
				}
				for _, v := range s.Updates {
					if v.WorkerID == a.WorkerID {
						expected = v.PrivateHash
					}
				}
				if expected == "" {
					t.Fatal("lost response preceded durable commit")
				}
				x.replaceRuntimeAttempt(t, i, expected)
				locals[i], closes[i] = relayIPCObserved(t, x.bindings[i], x.clients[i], scenario.observe)
				runtimes[i].Peer = locals[i]
				runtimes[i].Store, err = learner.Open(paths[i], 1<<20)
				check(t, err)
				runtimes[i].Scope.ExpectedPrivateHash = expected
			}
			first, err := runtimes[i].Execute(ctx, a)
			check(t, err)
			again, err := runtimes[i].Execute(ctx, a)
			if err != nil {
				t.Fatalf("round %d worker %d exact retry: %v", round, i, err)
			}
			if first != again || calls[i] != round+1 {
				t.Fatal("recovery repeated completed training", calls[i])
			}
		}
		for range 3 {
			check(t, x.model.Advance(ctx, x.s.Plan.Cohort.RunID))
		}
		r, err := locals[0].Poll(ctx)
		check(t, err)
		if r.Evaluation == nil {
			t.Fatal("missing local evaluation")
		}
		a := *r.Evaluation
		result, err := runtimes[0].Evaluate(ctx, a, scenario.evaluate)
		check(t, err)
		check(t, locals[0].Evaluate(ctx, result))
		check(t, locals[0].Evaluate(ctx, result))
		changed := result
		changed.Loss++
		if err = locals[0].Evaluate(ctx, changed); err == nil {
			t.Fatal("changed evaluation retry accepted")
		}
		if round == 0 {
			_, err = x.model.OpenRound(ctx, a.Binding.RunID, 600)
			check(t, err)
			check(t, locals[0].Evaluate(ctx, result))
			if err = locals[0].Evaluate(ctx, changed); err == nil {
				t.Fatal("changed historical evaluation accepted")
			}
		}
	}
	for i := range runtimes {
		var journal struct {
			Completed private.PrivateState
			Receipt   private.Receipt
		}
		b, err := paths[i].ReadFileLimit("head.json", 1<<20)
		check(t, err)
		check(t, json.Unmarshal(b, &journal))
		if journal.Completed.Replay.AttemptStep != 4 || journal.Receipt.PrivateHash == "" {
			t.Fatal("private replay did not continue")
		}
		scenario.verify(i, paths[i], journal.Completed)
		r, err := locals[i].Poll(ctx)
		check(t, err)
		if !r.Done {
			t.Fatal("local completion absent")
		}
		closes[i]()
	}
}
