//go:build darwin || linux

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/precision"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
)

// The production supervisor launches this test executable with its normal
// descriptor-only admission. Ordinary integration tests still run through m.
func TestMain(m *testing.M) {
	if len(os.Args) == 2 && os.Args[1] == "internal-worker-host" {
		if err := workerhost.ServeGuardian(os.NewFile(3, "guardian-control")); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[1] == "-mode" {
		if err := drainLearnerChild(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func drainLearnerChild() error {
	if len(os.Args) != 7 || os.Args[2] != "managed-worker" || os.Args[3] != "-worker-control-socket" || os.Args[5] != "-worker-session-fd" || os.Args[6] != "3" {
		return fmt.Errorf("unexpected managed learner launch")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	c, err := local.Connect(ctx, os.Args[4], os.NewFile(3, "session"), local.Limits{FrameBytes: wc.MaxFrameBytes, Bytes: 1 << 30, Messages: 100000})
	if err != nil {
		return err
	}
	defer func() { _ = c.Close() }()
	e, err := c.Receive(ctx)
	if err != nil {
		return err
	}
	a, err := workerjob.DecodeLearner(e.Payload, c.Binding())
	if err != nil {
		return err
	}
	d, err := local.NewDispatcher(ctx, c, local.DispatcherOptions{ReadTimeout: time.Minute})
	if err != nil {
		return err
	}
	defer func() { _ = d.Close() }()
	send := func(kind wc.Kind, event workerjob.Event) error {
		e, err := workerjob.Envelope(d.Binding(), 1, kind, event)
		if err != nil {
			return err
		}
		return d.Send(ctx, e)
	}
	if err := send(wc.KindReadiness, workerjob.Event{}); err != nil {
		return err
	}
	// No retry logic in this helper: HTTP outages must be absorbed by the real
	// agent client through the still-live supervisor and IPC session.
	calls, err := drainLearnerLoop(ctx, a, &learneripc.Client{Session: d})
	event := workerjob.Event{Committed: uint64(calls)}
	if err != nil {
		event.Error = err.Error()
	}
	if sendErr := send(wc.KindTerminalOutcome, event); err == nil {
		err = sendErr
	}
	return err
}

func drainLearnerLoop(ctx context.Context, a workerjob.LearnerAssignment, peer *learneripc.Client) (int, error) {
	var cfg struct {
		Clock int64 `json:"clock"`
	}
	if err := json.Unmarshal(a.Config, &cfg); err != nil {
		return 0, err
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: os.Getenv("MIXLAB_LEARNER_PRIVATE_DIR")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return 0, err
	}
	s, err := learner.Open(p, a.PrivateMaxBytes)
	if err != nil {
		return 0, err
	}
	calls := 0
	r := learner.Runtime{Store: s, Peer: peer, Now: func() int64 { return cfg.Clock },
		Scope: learner.Scope{RunID: a.RunID, PlanHash: a.PlanHash, CohortHash: a.Cohort.CohortHash, Generation: a.Cohort.Generation,
			WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, ParticipantID: a.Member.ParticipantID,
			IslandID: a.Member.IslandID, Contract: a.Contract, Coordinator: a.Coordinator, ExpectedPrivateHash: a.ExpectedPrivateHash}}
	r.Factory = func(model []optimizer.Tensor) (learner.Engine, error) {
		return &relayEngine{calls: &calls, state: learner.Snapshot{Model: relayClone(model), Optimizer: []byte("moment-0"), Sampler: []byte("cursor-0"), Schedule: []byte("schedule"), Metadata: []byte("metadata"),
			Replay: private.Replay{Version: "mixlab_learner_rng_v1", RunID: a.RunID, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, Seed: 7}, Precision: precision.Policy{ComputeDType: "float32", Backend: "metal"}}}, nil
	}
	first, err := peer.Poll(ctx)
	if err != nil {
		return calls, err
	}
	if first.Assignment == nil {
		return calls, fmt.Errorf("initial round absent")
	}
	for {
		next, err := peer.Poll(ctx)
		if err != nil {
			return calls, err
		}
		if next.Done {
			if calls != 1 {
				return calls, fmt.Errorf("trained %d times", calls)
			}
			return calls, nil
		}
		if next.Assignment != nil {
			if *next.Assignment != *first.Assignment {
				return calls, fmt.Errorf("restart changed round")
			}
			receipt, err := r.Execute(ctx, *next.Assignment)
			if err != nil {
				return calls, err
			}
			again, err := r.Execute(ctx, *next.Assignment)
			if err != nil || receipt != again {
				return calls, fmt.Errorf("receipt replay changed: %w", err)
			}
		}
		if next.Evaluation != nil {
			result, err := r.Evaluate(ctx, *next.Evaluation, func(context.Context, run.EvaluationAssignment, []optimizer.Tensor) (float64, uint64, error) {
				return 1, 8, nil
			})
			if err != nil {
				return calls, err
			}
			if err := peer.Evaluate(ctx, result); err != nil {
				return calls, err
			}
		}
		select {
		case <-ctx.Done():
			return calls, ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
}
