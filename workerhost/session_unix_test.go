//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerjob"
)

func learnerPlan(t *testing.T, mode string) (*Supervisor, LaunchPlan) {
	t.Helper()
	s, p := plan(t, mode)
	h := strings.Repeat("a", 64)
	member := distributed.LearnerMember{WorkerID: "worker", ParticipantID: "node", IslandID: "island", DatasetPartitionID: "partition"}
	cohort, err := distributed.NewLearnerCohort("run", 0, []distributed.LearnerMember{member})
	if err != nil {
		t.Fatal(err)
	}
	a := workerjob.LearnerAssignment{Version: workerjob.LearnerVersion, JobID: "job", AttemptID: "attempt", RunID: "run", PlanHash: h,
		Member: member, Cohort: &cohort, Contract: run.Contract{ConfigHash: h, ProgramHash: h, WeightLayoutHash: h, InnerOptimizerHash: h, DatasetHash: h, BuildHash: s.build},
		Coordinator: run.Coordinator{ClusterID: "cluster", PrincipalID: "coordinator", Role: "coordinator", Audience: "coordinator"},
		Config:      json.RawMessage(`{"opaque":true}`), DatasetSelector: mode, TrainPattern: "/unused/train.bin", RuntimeSeconds: 5, PrivateMaxBytes: 1 << 20}
	a.Contract.ConfigHash, err = workerjob.Digest(a.Config)
	if err != nil {
		t.Fatal(err)
	}
	p.Assignment = workerjob.Assignment{}
	p.Learner = &a
	p.LearnerPrivateDirectory = privateDirectory(t)
	p.Relay = func(_ context.Context, e wc.Envelope) (wc.Envelope, error) {
		e.Kind = wc.KindRoundResponse
		return e, nil
	}
	return s, p
}

func helperLearner(ctx context.Context, conn *local.Conn, initial wc.Envelope) int {
	a, err := workerjob.DecodeLearner(initial.Payload, conn.Binding())
	if err != nil || a.Cohort == nil {
		return 60
	}
	if os.Getenv("MLX_HOSTFILE") != "" || os.Getenv("MLX_RANK") != "" || os.Getenv("WORKERHOST_SECRET") != "" {
		return 61
	}
	if _, err := os.Stat("ring.json"); !os.IsNotExist(err) {
		return 62
	}
	private, err := statehome.Resolve(statehome.Options{ExactDir: os.Getenv("MIXLAB_LEARNER_PRIVATE_DIR")}, statehome.Context{Kind: statehome.Worker})
	if err != nil || private.Validate() != nil {
		return 63
	}
	if a.DatasetSelector == "runtime-store-startup" {
		return helperManagedLearnerRuntime(ctx, conn, a, private)
	}
	if a.ExpectedPrivateHash != "" {
		if body, err := os.ReadFile(filepath.Join(private.Dir(), "retained")); err != nil || string(body) != "state" {
			return 64
		}
	} else if private.WriteFile("retained", []byte("state")) != nil {
		return 65
	}
	d, err := local.NewDispatcher(ctx, conn, local.DispatcherOptions{})
	if err != nil {
		return 66
	}
	defer func() { _ = d.Close() }()
	send := func(kind wc.Kind) error {
		e, err := workerjob.Envelope(d.Binding(), 1, kind, workerjob.Event{})
		if err != nil {
			return err
		}
		return d.Send(ctx, e)
	}
	if a.DatasetSelector == "startup-before-relay" {
		<-d.Done()
		return 73
	}
	if a.DatasetSelector != "no-ready" {
		if send(wc.KindReadiness) != nil {
			return 67
		}
	}
	request := wc.Envelope{Kind: wc.KindRoundRequest, PayloadKind: "owner_opaque_v99", PayloadVersion: 99, Payload: []byte(`{"unknown_to_host":[1,2,3]}`)}
	result := make(chan error, 1)
	go func() {
		e, err := d.Request(ctx, request)
		if err == nil && (e.Kind != wc.KindRoundResponse || string(e.Payload) != string(request.Payload)) {
			err = fmt.Errorf("opaque reply changed")
		}
		result <- err
	}()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case err := <-result:
			if err != nil {
				return 68
			}
			if send(wc.KindTerminalOutcome) != nil {
				return 69
			}
			return 0
		case <-ticker.C:
			if send(wc.KindHeartbeat) != nil {
				return 70
			}
			if a.DatasetSelector == "cancel" {
				// Test-only handshake: cancel an active relay, never race the
				// request goroutine before its first frame has been admitted.
				if _, err := os.Stat("relay-started"); os.IsNotExist(err) {
					continue
				}
				_ = send(wc.KindCancellation)
				return 71
			}
		case <-ctx.Done():
			return 72
		}
	}
}

func TestLearnerLaunchOpaqueRelayAndPrivateRetry(t *testing.T) {
	t.Setenv("MLX_HOSTFILE", "must-not-inherit")
	t.Setenv("MLX_RANK", "42")
	t.Setenv("WORKERHOST_SECRET", "must-not-inherit")
	s, p := learnerPlan(t, "success")
	var calls atomic.Int32
	p.Relay = func(ctx context.Context, e wc.Envelope) (wc.Envelope, error) {
		calls.Add(1)
		if e.PayloadKind != "owner_opaque_v99" || e.PayloadVersion != 99 {
			return wc.Envelope{}, fmt.Errorf("changed opaque routing")
		}
		select {
		case <-time.After(150 * time.Millisecond):
		case <-ctx.Done():
			return wc.Envelope{}, ctx.Err()
		}
		e.Kind = wc.KindRoundResponse
		return e, nil
	}
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 {
			p.Directory = privateDirectory(t)
			p.Learner.AttemptID = "retry"
			p.Learner.ExpectedPrivateHash = strings.Repeat("a", 64)
		}
		before, err := p.Learner.Binding()
		if err != nil {
			t.Fatal(err)
		}
		r, err := s.Run(context.Background(), p)
		if err != nil || !r.Ready || r.PID == 0 {
			log, _ := os.ReadFile(filepath.Join(p.Directory.Dir(), "worker.log"))
			t.Fatalf("learner result %+v: %v %s", r, err, log)
		}
		after, err := p.Learner.Binding()
		if err != nil || before != after {
			t.Fatal("launch changed assignment")
		}
		if _, err := os.Stat(filepath.Join(p.Directory.Dir(), "ring.json")); !os.IsNotExist(err) {
			t.Fatalf("learner got ring: %v", err)
		}
	}
	if calls.Load() != 2 {
		t.Fatalf("relay calls %d", calls.Load())
	}
}

func TestLearnerRelayDeadlineCancellationAndStartup(t *testing.T) {
	for _, mode := range []string{"timeout", "cancel", "no-ready", "startup-before-relay", "relay-error"} {
		t.Run(mode, func(t *testing.T) {
			s, p := learnerPlan(t, mode)
			p.Learner.RuntimeSeconds = 10
			p.StartupTimeout = 3 * time.Second
			p.RelayTimeout = 3 * time.Second
			if mode == "timeout" {
				p.RelayTimeout = 500 * time.Millisecond
			}
			startup := mode == "no-ready" || mode == "startup-before-relay"
			if startup {
				p.StartupTimeout = 300 * time.Millisecond
			}
			callbackStarted := make(chan struct{})
			callbackDone := make(chan struct{})
			relayFailure := errors.New("relay failure")
			p.Relay = func(ctx context.Context, _ wc.Envelope) (wc.Envelope, error) {
				close(callbackStarted)
				defer close(callbackDone)
				if mode == "relay-error" {
					return wc.Envelope{}, relayFailure
				}
				if mode == "cancel" {
					if err := p.Directory.WriteFile("relay-started", []byte("ready")); err != nil {
						return wc.Envelope{}, err
					}
				}
				<-ctx.Done()
				return wc.Envelope{}, ctx.Err()
			}
			ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
			defer cancel()
			start := time.Now()
			r, err := s.Run(ctx, p)
			if err == nil || r.PID == 0 || time.Since(start) > 8*time.Second {
				t.Fatalf("unbounded relay %+v %v", r, err)
			}
			if mode == "relay-error" && !errors.Is(err, relayFailure) {
				t.Fatalf("relay callback error not returned: result=%+v err=%v", r, err)
			}
			if (startup || mode == "timeout") && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("expected deadline: result=%+v err=%v", r, err)
			}
			if startup && r.Ready {
				t.Fatal("startup timeout after readiness")
			}
			select {
			case <-callbackStarted:
				if mode == "startup-before-relay" {
					t.Fatal("callback ran before any request")
				}
			default:
				// Startup may expire before the child submits a request. There
				// is then no callback to cancel, which is a valid startup failure.
				if startup {
					return
				}
				t.Fatalf("relay never started: result=%+v err=%v", r, err)
			}
			select {
			case <-callbackDone:
			case <-time.After(time.Second):
				t.Fatalf("active relay did not finish: result=%+v err=%v", r, err)
			}
		})
	}
}

func TestLearnerLaunchRejectsInvalidOwnership(t *testing.T) {
	for _, mode := range []string{"mixed-schema", "no-relay", "no-cohort", "shared-directory", "nested-directory", "missing-initial", "missing-retry", "wrong-kind", "negative-timeout"} {
		t.Run(mode, func(t *testing.T) {
			s, p := learnerPlan(t, mode)
			switch mode {
			case "mixed-schema":
				p.Assignment.Version = workerjob.Version
			case "no-relay":
				p.Relay = nil
			case "no-cohort":
				p.Learner.Cohort = nil
			case "shared-directory":
				p.LearnerPrivateDirectory = p.Directory
			case "nested-directory":
				var err error
				p.LearnerPrivateDirectory, err = statehome.Resolve(statehome.Options{ExactDir: filepath.Join(p.Directory.Dir(), "private")}, statehome.Context{Kind: statehome.Worker})
				if err != nil {
					t.Fatal(err)
				}
			case "missing-initial", "missing-retry":
				if err := os.Remove(p.LearnerPrivateDirectory.Dir()); err != nil {
					t.Fatal(err)
				}
				if mode == "missing-retry" {
					p.Learner.ExpectedPrivateHash = strings.Repeat("a", 64)
				}
			case "wrong-kind":
				p.LearnerPrivateDirectory = statehome.Path{}
			case "negative-timeout":
				p.RelayTimeout = -time.Second
			}
			r, err := s.Run(context.Background(), p)
			if err == nil || r.PID != 0 {
				t.Fatalf("invalid launch: %+v %v", r, err)
			}
			if mode == "missing-initial" || mode == "missing-retry" {
				if _, err := os.Stat(p.LearnerPrivateDirectory.Dir()); !errors.Is(err, os.ErrNotExist) {
					t.Fatal("retry created missing private state")
				}
			}
		})
	}
}
