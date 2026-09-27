package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

// NodeExecutionPorts are local composition ports. Open resolves only private
// job/attempt state; Authorize checks current trust rather than saved admission.
// Cleanup must stop and join the relay before destroying its workload key.
type NodeExecutionPorts struct {
	Open      func(context.Context, string, string) (*workerhost.AttemptStore, workerhost.Runner, error)
	Authorize func(context.Context, nodeagent.LocalExecution) error
	Cleanup   func(context.Context, nodeagent.CleanupRequest) error
	Clock     func() time.Time
	Ready     func()
	Event     func(error)
}

type nodeExecutionRun struct {
	job    string
	cancel context.CancelFunc
	done   chan error
}

type nodeExecutionLoop struct {
	store   *nodeagent.Store
	ports   NodeExecutionPorts
	running *nodeExecutionRun
}

// RunNodeExecution owns one exclusive node runtime. Restart cancels old intent;
// it never relaunches an interrupted job. Ready runs only after that durable
// restart fence, so the server must not accept mutations before Ready.
func RunNodeExecution(ctx context.Context, store *nodeagent.Store, ports NodeExecutionPorts) error {
	if store == nil || ports.Open == nil || ports.Authorize == nil || ports.Cleanup == nil || ports.Clock == nil || ports.Ready == nil {
		return fmt.Errorf("node execution store, local ports and readiness callback required")
	}
	return store.OwnExecution(ctx, func(ctx context.Context) (result error) {
		if err := store.InterruptActive(ctx, "agent_restart"); err != nil {
			return err
		}
		life, cancel := context.WithCancel(ctx)
		defer cancel()
		loop := &nodeExecutionLoop{store: store, ports: ports}
		defer func() {
			cancel()
			cleanupCtx, done := context.WithTimeout(context.Background(), 45*time.Second)
			defer done()
			result = errors.Join(result, loop.shutdown(cleanupCtx))
		}()
		if err := loop.poll(life); err != nil && !errors.Is(err, workerhost.ErrReconciliationRequired) {
			if life.Err() != nil && errors.Is(err, life.Err()) {
				return nil // Deferred shutdown uses a fresh cleanup deadline.
			}
			return err
		}
		ports.Ready()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return nil
			case <-ticker.C:
				if err := loop.poll(life); err != nil {
					if life.Err() != nil && errors.Is(err, life.Err()) {
						return nil
					}
					if ports.Event != nil {
						ports.Event(err)
					}
					if !errors.Is(err, workerhost.ErrReconciliationRequired) {
						return err
					}
				}
			}
		}
	})
}

func (l *nodeExecutionLoop) poll(ctx context.Context) error {
	if l.running != nil {
		select {
		case err := <-l.running.done:
			l.running.cancel()
			l.running = nil
			if errors.Is(err, workerhost.ErrReconciliationRequired) {
				if err := l.store.InterruptActive(ctx, "agent_shutdown"); err != nil {
					return err
				}
			}
			if err != nil && !errors.Is(err, workerhost.ErrReconciliationRequired) {
				return err
			}
		default:
		}
	}
	if err := l.store.Expire(ctx, l.ports.Clock()); err != nil {
		return err
	}
	x, err := l.store.ActiveExecution(ctx)
	if err != nil || x == nil {
		return err
	}
	if x.Lease.State != nodeagent.Releasing {
		if err := l.ports.Authorize(ctx, *x); err != nil {
			if l.ports.Event != nil {
				l.ports.Event(fmt.Errorf("execution authority unavailable: %w", err))
			}
			if err := l.store.InterruptActive(ctx, "authority_unavailable"); err != nil {
				return err
			}
			x, err = l.store.ActiveExecution(ctx)
			if err != nil || x == nil {
				return err
			}
		}
	}
	stopping := x.Lease.State == nodeagent.Releasing || (x.Job != nil && x.Job.CancelRequested)
	if l.running != nil {
		if x.Job == nil || l.running.job != x.Job.ID {
			return fmt.Errorf("running job lost its exclusive lease")
		}
		if stopping {
			l.running.cancel()
		}
		return nil
	}
	if x.Job == nil {
		if stopping {
			return l.store.ReleaseUnprepared(ctx, x.Lease.ID)
		}
		return nil
	}
	if x.Outcome != nil && x.Outcome.Terminal() {
		// Replaying the outcome repairs job-first/lease-second publication.
		if _, err := l.store.ApplyOutcome(ctx, *x.Outcome); err != nil {
			return err
		}
		return l.store.CompleteCleanup(ctx, x.Job.ID, l.ports.Cleanup)
	}
	if !stopping && x.Approval == nil {
		return nil
	}
	host, runner, err := l.ports.Open(ctx, x.Job.ID, x.Job.Attempt)
	if err != nil {
		return err
	}
	if host == nil || runner == nil {
		return fmt.Errorf("incomplete local execution ports")
	}
	if stopping {
		return l.reconcile(ctx, *x, host, runner)
	}
	life, cancel := context.WithCancel(ctx)
	run := &nodeExecutionRun{job: x.Job.ID, cancel: cancel, done: make(chan error, 1)}
	l.running = run
	approval := *x.Approval
	go func() {
		out, err := host.Run(life, approval, nodeOutcomeRunner{Runner: runner, host: host, store: l.store})
		if err == nil {
			publishCtx, done := context.WithTimeout(context.Background(), 5*time.Second)
			_, err = l.store.ApplyOutcome(publishCtx, out)
			done()
		}
		run.done <- err
	}()
	return nil
}

func (l *nodeExecutionLoop) reconcile(ctx context.Context, x nodeagent.LocalExecution, host *workerhost.AttemptStore, runner workerhost.Runner) error {
	var out contract.Outcome
	var err error
	if x.Approval == nil {
		fence, e := l.store.PreparationFence(ctx, x.Job.ID)
		if e != nil {
			return e
		}
		out, err = host.FencePreparation(ctx, fence)
	} else {
		out, err = host.FenceApproved(ctx, *x.Approval)
		if errors.Is(err, workerhost.ErrReconciliationRequired) {
			out, err = host.Reconcile(ctx, runner)
		}
	}
	if err != nil {
		return err
	}
	if _, err := l.store.ApplyOutcome(ctx, out); err != nil {
		return err
	}
	return l.store.CompleteCleanup(ctx, x.Job.ID, l.ports.Cleanup)
}

func (l *nodeExecutionLoop) shutdown(ctx context.Context) error {
	if l.running != nil {
		l.running.cancel()
	}
	if err := l.store.InterruptActive(ctx, "agent_shutdown"); err != nil {
		return err
	}
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		err := l.poll(ctx)
		if err != nil && !errors.Is(err, workerhost.ErrReconciliationRequired) {
			return err
		}
		x, readErr := l.store.ActiveExecution(ctx)
		if readErr != nil {
			return readErr
		}
		if x == nil && l.running == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return errors.Join(ctx.Err(), workerhost.ErrReconciliationRequired)
		case <-ticker.C:
		}
	}
}

type nodeOutcomeRunner struct {
	workerhost.Runner
	host  *workerhost.AttemptStore
	store *nodeagent.Store
}

func (r nodeOutcomeRunner) Run(ctx context.Context, a contract.Approved, started func(int) error) error {
	return r.Runner.Run(ctx, a, func(pid int) error {
		if err := started(pid); err != nil {
			return err
		}
		s, err := r.host.Status()
		if err != nil {
			return err
		}
		if s.Started == nil {
			return fmt.Errorf("hosting start evidence missing")
		}
		_, err = r.store.ApplyOutcome(ctx, *s.Started)
		return err
	})
}
