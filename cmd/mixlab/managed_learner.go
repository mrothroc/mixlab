package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/learner"
	"github.com/mrothroc/mixlab/learneripc"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/train"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workercontrol/local"
	"github.com/mrothroc/mixlab/workerjob"
)

func serveManagedLearner(parent context.Context, s *local.Dispatcher, e wc.Envelope) error {
	a, err := workerjob.DecodeLearner(e.Payload, s.Binding())
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	id, err := workerjob.FileDigest(exe)
	if err != nil || id != a.Contract.BuildHash {
		return fmt.Errorf("learner executable differs from approved build")
	}
	ctx, cancel := context.WithTimeout(parent, time.Duration(a.RuntimeSeconds)*time.Second)
	defer cancel()
	send := func(kind wc.Kind, event workerjob.Event) error {
		msg, err := workerjob.Envelope(s.Binding(), 1, kind, event)
		if err != nil {
			return err
		}
		return s.Send(ctx, msg)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		select {
		case <-s.Controls():
		case <-s.Events():
		case <-s.Done():
		case <-ctx.Done():
		}
		cancel()
	}()
	go func() {
		defer wg.Done()
		timer := time.NewTicker(5 * time.Second)
		defer timer.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-timer.C:
				if send(wc.KindHeartbeat, workerjob.Event{}) != nil {
					cancel()
					return
				}
			}
		}
	}()
	defer func() { cancel(); wg.Wait() }()
	err = runManagedLearner(ctx, a, &learneripc.Client{Session: s}, send)
	terminal := workerjob.Event{}
	if err != nil {
		terminal.Error = "managed learner failed: " + err.Error()
		if len(terminal.Error) > 4096 {
			terminal.Error = "managed learner failed"
		}
	}
	if e := send(wc.KindTerminalOutcome, terminal); err == nil {
		err = e
	}
	return err
}

func runManagedLearner(ctx context.Context, a workerjob.LearnerAssignment, peer *learneripc.Client, send func(wc.Kind, workerjob.Event) error) error {
	cfg, identity, err := train.PrepareLocalLearner(a)
	if err != nil {
		return err
	}
	dir := os.Getenv("MIXLAB_LEARNER_PRIVATE_DIR")
	if dir == "" {
		return fmt.Errorf("hosting did not supply a private learner directory")
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return err
	}
	store, err := learner.Open(p, a.PrivateMaxBytes)
	if err != nil {
		return err
	}
	r := learner.Runtime{Store: store, Peer: peer, Scope: learner.Scope{RunID: a.RunID, PlanHash: a.PlanHash, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, ParticipantID: a.Member.ParticipantID, IslandID: a.Member.IslandID, CohortHash: a.Cohort.CohortHash, Generation: a.Cohort.Generation, Contract: a.Contract, Coordinator: a.Coordinator, ExpectedPrivateHash: a.ExpectedPrivateHash}}
	r.Factory = func(v []optimizer.Tensor) (learner.Engine, error) {
		model, err := train.LocalLearnerModelFromWire(v)
		if err != nil {
			return nil, err
		}
		s, err := train.NewLocalLearnerSession(cfg, a.TrainPattern, identity, model)
		if err != nil {
			return nil, err
		}
		return &learnerEngine{session: s}, nil
	}
	if err = send(wc.KindReadiness, workerjob.Event{}); err != nil {
		return err
	}
	for {
		if err = ctx.Err(); err != nil {
			return err
		}
		next, e := peer.Poll(ctx)
		if e != nil {
			return e
		}
		if next.Done {
			return nil
		}
		if next.Assignment != nil {
			if _, err = r.Execute(ctx, *next.Assignment); err != nil {
				return err
			}
			step := (next.Assignment.Round + 1) * next.Assignment.InnerSteps
			if err = send(wc.KindProgress, workerjob.Event{Step: int(step), Committed: step}); err != nil {
				return err
			}
		}
		if next.Evaluation != nil {
			if a.Validation == nil {
				return fmt.Errorf("evaluation requires a locally configured validation dataset")
			}
			if err := a.Validation.Match(*next.Evaluation); err != nil {
				return err
			}
			result, err := r.Evaluate(ctx, *next.Evaluation, func(ctx context.Context, assignment run.EvaluationAssignment, model []optimizer.Tensor) (float64, uint64, error) {
				local, err := train.LocalLearnerModelFromWire(model)
				if err != nil {
					return 0, 0, err
				}
				return train.EvaluateLocalLearnerModel(ctx, cfg, local, *a.Validation, assignment.Precision)
			})
			if err != nil {
				return err
			}
			if err := peer.Evaluate(ctx, result); err != nil {
				return err
			}
		}
		timer := time.NewTimer(200 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}
