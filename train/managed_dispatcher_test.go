//go:build darwin || linux

package train

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerjob"
)

func managedConnectionPair(t *testing.T, binding wc.Binding) (*local.Conn, *local.Dispatcher) {
	t.Helper()
	limits := local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: workerjob.ControlByteBudget, Messages: workerjob.ControlMessageBudget}
	server, credential, err := local.Listen(managedPrivateDir(t), limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Close() })
	fd, err := syscall.Dup(int(credential.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	childCredential := os.NewFile(uintptr(fd), "test-child-session")
	t.Cleanup(func() { _ = childCredential.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	type connected struct {
		conn *local.Conn
		err  error
	}
	accepted := make(chan connected, 1)
	go func() {
		c, err := server.Accept(ctx, binding, wc.PeerIdentity{PID: os.Getpid(), UID: uint32(os.Getuid())})
		accepted <- connected{c, err}
	}()
	child, err := local.Connect(ctx, server.Path(), childCredential, limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = child.Close() })
	parent := <-accepted
	if parent.err != nil {
		t.Fatal(parent.err)
	}
	d, err := local.NewDispatcher(context.Background(), parent.conn, local.DispatcherOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return child, d
}

func TestManagedDDPDispatcherCancellation(t *testing.T) {
	for _, mode := range []string{"cancel", "unexpected", "disconnect", "success"} {
		t.Run(mode, func(t *testing.T) {
			a := managedFixture(t)
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			a.BuildID, err = workerjob.FileDigest(exe)
			if err != nil {
				t.Fatal(err)
			}
			binding, err := a.Binding()
			if err != nil {
				t.Fatal(err)
			}
			child, host := managedConnectionPair(t, binding)
			started := make(chan struct{})
			run := func(ctx context.Context, _ workerjob.Assignment, ready func() error, progress func(workerjob.Event) error) error {
				if err := ready(); err != nil {
					return err
				}
				if err := progress(workerjob.Event{Step: 1, Committed: 1}); err != nil {
					return err
				}
				close(started)
				if mode == "success" {
					return nil
				}
				<-ctx.Done()
				return ctx.Err()
			}
			result := make(chan error, 1)
			go func() { result <- serveManagedWorker(context.Background(), child, run) }()
			initial, err := workerjob.Envelope(binding, 1, wc.KindAssignment, a)
			if err != nil {
				t.Fatal(err)
			}
			if err := host.Send(context.Background(), initial); err != nil {
				t.Fatal(err)
			}
			select {
			case <-started:
			case <-time.After(3 * time.Second):
				t.Fatal("DDP did not start")
			}
			switch mode {
			case "disconnect":
				_ = host.Close()
			case "cancel", "unexpected":
				kind := wc.KindCancellation
				if mode == "unexpected" {
					kind = wc.KindProgress
				}
				e, err := workerjob.Envelope(binding, 1, kind, workerjob.Event{})
				if err != nil {
					t.Fatal(err)
				}
				if err := host.Send(context.Background(), e); err != nil {
					t.Fatal(err)
				}
			}
			select {
			case err := <-result:
				if (err == nil) != (mode == "success") {
					t.Fatalf("mode %s: %v", mode, err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("managed DDP did not stop")
			}
			if mode == "success" {
				var kinds []wc.Kind
				for e := range host.Events() {
					kinds = append(kinds, e.Kind)
				}
				if fmt.Sprint(kinds) != "[readiness progress terminal_outcome]" {
					t.Fatalf("lost lifecycle: %v", kinds)
				}
			}
		})
	}
}

func managedLearnerAssignment(t *testing.T) (wc.Binding, wc.Envelope) {
	t.Helper()
	h := strings.Repeat("a", 64)
	member := distributed.LearnerMember{WorkerID: "worker", ParticipantID: "node", IslandID: "island", DatasetPartitionID: "partition"}
	cohort, err := distributed.NewLearnerCohort("run", 0, []distributed.LearnerMember{member})
	if err != nil {
		t.Fatal(err)
	}
	a := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: "job", AttemptID: "attempt", RunID: "run", PlanHash: h,
		Member: member, Cohort: &cohort, Contract: run.Contract{ConfigHash: h, ProgramHash: h, WeightLayoutHash: h, InnerOptimizerHash: h, DatasetHash: h, BuildHash: h},
		Coordinator: run.Coordinator{ClusterID: "cluster", PrincipalID: "coordinator", Role: "coordinator", Audience: "coordinator"},
		Config:      json.RawMessage(`{"owner":"opaque"}`), DatasetSelector: "fixture", TrainPattern: "/unused/train.bin", RuntimeSeconds: 5, PrivateMaxBytes: 1024}
	a.Contract.ConfigHash, err = workerjob.Digest(a.Config)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := a.Binding()
	if err != nil {
		t.Fatal(err)
	}
	payload, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	return binding, wc.Envelope{Kind: wc.KindAssignment, CorrelationID: binding.AttemptID, PayloadKind: workerjob.LearnerVersion, PayloadVersion: 1, Payload: payload}
}

func TestManagedDispatcherSelectsComposedSchema(t *testing.T) {
	for _, mode := range []string{"learner", "not-composed", "wrong-kind", "wrong-correlation", "missing-correlation", "payload-version", "payload-kind", "ddp-schema", "future-schema", "missing-schema", "null-schema", "case-alias-schema", "non-object", "unknown-field", "assignment-binding"} {
		t.Run(mode, func(t *testing.T) {
			binding, e := managedLearnerAssignment(t)
			original := append([]byte(nil), e.Payload...)
			child, host := managedConnectionPair(t, binding)
			ownerResult := errors.New("owner callback result")
			var learnerCalls, ddpCalls atomic.Int32
			callback := func(_ context.Context, d *local.Dispatcher, e wc.Envelope) error {
				learnerCalls.Add(1)
				if d.Binding() != binding || !bytes.Equal(e.Payload, original) {
					return fmt.Errorf("callback input changed")
				}
				return ownerResult
			}
			if mode == "not-composed" {
				callback = nil
			}
			run := func(context.Context, workerjob.Assignment, func() error, func(workerjob.Event) error) error {
				ddpCalls.Add(1)
				return fmt.Errorf("unexpected DDP call")
			}
			result := make(chan error, 1)
			go func() { result <- serveManagedWorkerWithLearner(context.Background(), child, run, callback) }()
			switch mode {
			case "wrong-kind":
				e.Kind = wc.KindProgress
			case "wrong-correlation":
				e.CorrelationID = "other-attempt"
			case "missing-correlation":
				e.CorrelationID = "" // dispatcher generates an ID, not the required attempt ID
			case "payload-version":
				e.PayloadVersion = 2
			case "payload-kind":
				e.PayloadKind = workerjob.Version
			case "ddp-schema":
				e.Payload = bytes.Replace(e.Payload, []byte(workerjob.LearnerVersion), []byte(workerjob.Version), 1)
			case "future-schema":
				e.Payload = bytes.Replace(e.Payload, []byte(workerjob.LearnerVersion), []byte("mixlab_local_learner_assignment_v2"), 1)
			case "missing-schema":
				e.Payload = bytes.Replace(e.Payload, []byte(`"version":"`+workerjob.LearnerVersion+`",`), nil, 1)
			case "null-schema":
				e.Payload = bytes.Replace(e.Payload, []byte(`"version":"`+workerjob.LearnerVersion+`"`), []byte(`"version":null`), 1)
			case "case-alias-schema":
				e.Payload = bytes.Replace(e.Payload, []byte(`"version":`), []byte(`"Version":`), 1)
			case "non-object":
				e.Payload = []byte(`null`)
			case "unknown-field":
				e.Payload = append([]byte(`{"credential":"not-allowed",`), e.Payload[1:]...)
			case "assignment-binding":
				e.Payload = bytes.Replace(e.Payload, []byte(`"fixture"`), []byte(`"different-fixture"`), 1)
			}
			if err := host.Send(context.Background(), e); err != nil {
				t.Fatal(err)
			}
			select {
			case err := <-result:
				if err == nil || errors.Is(err, ownerResult) != (mode == "learner") {
					t.Fatalf("dispatch: %v", err)
				}
				wantCalls := int32(0)
				if mode == "learner" {
					wantCalls = 1
				}
				if learnerCalls.Load() != wantCalls || ddpCalls.Load() != 0 {
					t.Fatalf("invalid callback dispatch: learner=%d ddp=%d", learnerCalls.Load(), ddpCalls.Load())
				}
			case <-time.After(3 * time.Second):
				t.Fatal("dispatch blocked")
			}
		})
	}
}
