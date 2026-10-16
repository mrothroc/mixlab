package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost"
)

func (l *nodeExecutionLoop) cleanupLearner(ctx context.Context, x nodeagent.LocalExecution) error {
	return l.store.CleanupLearner(ctx, x.Job.ID, func(ctx context.Context, scope workload.Scope) error {
		return l.ports.Cleanup(ctx, nodeagent.CleanupRequest{Job: x.Job.ID, Attempt: scope.Attempt, Workload: scope, WorkloadInitialized: true})
	})
}

func (l *nodeExecutionLoop) pollLearner(ctx context.Context, x nodeagent.LocalExecution, stopping bool) error {
	if x.Outcome != nil && x.Outcome.Terminal() {
		if err := l.store.ApplyLearnerOutcome(ctx, *x.Outcome); err != nil {
			return err
		}
		if stopping {
			return l.cleanupLearner(ctx, x)
		}
		return nil // Keep the original allocation for explicit same-worker recovery.
	}
	if x.Approval == nil {
		if stopping {
			return l.cleanupLearner(ctx, x)
		}
		return nil
	}
	if x.Learner.Stopped && !stopping {
		return nil
	}
	if l.ports.OpenLearner == nil {
		return fmt.Errorf("managed learner runtime not configured")
	}
	host, runner, err := l.ports.OpenLearner(ctx, x)
	if err != nil {
		return err
	}
	if host == nil || runner == nil {
		return fmt.Errorf("incomplete learner hosting ports")
	}
	if stopping {
		out, err := host.FenceApproved(ctx, *x.Approval)
		if errors.Is(err, workerhost.ErrReconciliationRequired) {
			out, err = host.Reconcile(ctx, runner)
		}
		if err != nil {
			return err
		}
		if err := l.store.ApplyLearnerOutcome(ctx, out); err != nil {
			return err
		}
		return l.cleanupLearner(ctx, x)
	}
	life, cancel := context.WithCancel(ctx)
	run := &nodeExecutionRun{job: x.Job.ID, cancel: cancel, done: make(chan error, 1)}
	l.running = run
	go func() {
		out, err := host.Run(life, *x.Approval, nodeOutcomeRunner{Runner: runner, host: host, store: l.store, learner: true})
		if err == nil {
			publish, done := context.WithTimeout(context.Background(), 5*time.Second)
			err = l.store.ApplyLearnerOutcome(publish, out)
			done()
		}
		run.done <- err
	}()
	return nil
}
