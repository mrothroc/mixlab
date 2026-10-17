package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust/workload"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

func (l *nodeExecutionLoop) cleanupLearner(ctx context.Context, x nodeagent.LocalExecution) error {
	return l.store.CleanupLearner(ctx, x.Job.ID, func(ctx context.Context, scope workload.Scope) error {
		return l.ports.Cleanup(ctx, nodeagent.CleanupRequest{Job: x.Job.ID, Attempt: scope.Attempt, Workload: scope, WorkloadInitialized: true})
	})
}

func (l *nodeExecutionLoop) pollLearner(ctx context.Context, x nodeagent.LocalExecution, stopping bool) error {
	if x.Outcome != nil && x.Outcome.Terminal() {
		// ActiveExecution already validated this durable outcome. A retry may
		// have replaced its attempt since that read; do not republish it.
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
			l.logLearnerExit(*x.Approval, out, err)
			return err
		}
		l.logLearnerExit(*x.Approval, out, nil)
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
		l.logLearnerExit(*x.Approval, out, err)
		if err == nil {
			publish, done := context.WithTimeout(context.Background(), 5*time.Second)
			err = l.store.ApplyLearnerOutcome(publish, out)
			done()
		}
		run.done <- err
	}()
	return nil
}

// Emit local diagnostics even when hosting cannot publish a terminal outcome.
// JSON quoting prevents child-controlled errors from forging extra log lines.
func (l *nodeExecutionLoop) logLearnerExit(a contract.Approved, o contract.Outcome, hostErr error) {
	if l.ports.Event == nil {
		return
	}
	now := time.Now()
	if l.ports.Clock != nil {
		now = l.ports.Clock()
	}
	record := struct {
		Time    string            `json:"time"`
		Job     string            `json:"job"`
		Attempt string            `json:"attempt"`
		Outcome *contract.Outcome `json:"outcome,omitempty"`
		Error   string            `json:"hosting_error,omitempty"`
	}{Time: now.UTC().Format(time.RFC3339Nano), Job: a.JobID(), Attempt: a.AttemptID()}
	if o.Validate() == nil && o.Terminal() {
		record.Outcome = &o
	}
	if hostErr != nil {
		record.Error = hostErr.Error()
		if len(record.Error) > 4096 {
			record.Error = record.Error[:4096]
		}
	}
	b, _ := json.Marshal(record)
	l.ports.Event(fmt.Errorf("learner exit: %s", b))
}
