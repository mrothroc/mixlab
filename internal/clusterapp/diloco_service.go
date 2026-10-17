package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/admission"
)

// DiLoCoService composes lifecycle, not numerical or admission decisions.
// Each callback must reconstruct its owner from the accepted immutable intent.
// Work runs under the service lifetime, never an HTTP submission's context.
type DiLoCoService struct {
	Inbox     *admission.Inbox
	Admission func(context.Context, admission.Submission) (admission.Ports, error)
	Maintain  func(context.Context, admission.Submission, admission.Snapshot) error
	Run       func(context.Context, admission.Submission, admission.Snapshot) error
	Clock     func() time.Time
	Interval  time.Duration
	Event     func(string, error)
}

func (s DiLoCoService) validate() error {
	if s.Inbox == nil || s.Admission == nil || s.Run == nil || s.Clock == nil || s.Interval < 10*time.Millisecond || s.Interval > time.Minute {
		return fmt.Errorf("bounded durable coordinator service ports required")
	}
	return nil
}

// step advances one bounded effect. A run's next step never overlaps its prior
// step; different runs have independent service-owned lifetimes.
func (s DiLoCoService) step(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	q, err := s.Inbox.Read(ctx, id)
	if err != nil {
		return err
	}
	saga, err := s.Inbox.Resume(ctx, id)
	if err != nil {
		return err
	}
	v, err := saga.Snapshot(ctx)
	if err != nil {
		return err
	}
	if v.Done && !v.Committed {
		return nil
	}
	budget := 30 * time.Second
	if v.Committed {
		// Includes lease observations, renewal and status/recovery work. Each
		// authenticated operation has its own shorter bound; the service turn
		// must not truncate them solely because the host is slow.
		budget = 3 * time.Minute
	}
	work, cancel := context.WithTimeout(ctx, budget)
	var effect error
	if !v.Done {
		var maintenance error
		if s.Maintain != nil && (!v.Aborting || v.Committed) {
			maintenance = s.Maintain(work, q, v)
			if errors.Is(maintenance, ErrAdmissionLeaseLost) && !v.Committed {
				// Fence forward admission before compensating. A transient network
				// error instead waits for a fresh observation on the next tick.
				if err := saga.Cancel(work); err != nil {
					cancel()
					return err
				}
				v.Aborting = true
			}
		}
		var p admission.Ports
		// Once committed, retry handoff even when maintenance cannot reach one
		// node. Compensation and handoff must never be gated by renewal failure.
		if maintenance == nil || v.Aborting || v.Committed {
			p, effect = s.Admission(work, q)
			if effect == nil {
				_, effect = saga.Step(work, p)
			}
		}
		effect = errors.Join(maintenance, effect)
	} else {
		effect = s.Run(work, q, v)
	}
	cancel()
	if ctx.Err() != nil {
		return ctx.Err()
	} // Shutdown never requests compensation.
	if effect != nil && len(effect.Error()) > 4096 {
		effect = fmt.Errorf("service effect failed; diagnostic exceeded limit")
	}
	// Observation is replaceable operational evidence, not the local event log.
	// A failed observation write must not hide the original effect diagnostic.
	if effect != nil && s.Event != nil {
		s.Event(id, effect)
	}
	return s.Inbox.Observe(ctx, id, s.Clock().Unix(), effect)
}

func (s DiLoCoService) Serve(ctx context.Context, ready func()) error {
	if err := s.validate(); err != nil {
		return err
	}
	return s.Inbox.OwnService(ctx, func(ctx context.Context) error {
		life, cancel := context.WithCancel(ctx)
		var wg sync.WaitGroup
		defer func() { cancel(); wg.Wait() }()
		// The durable inbox caps retained runs at 1024. One actor per accepted
		// run avoids head-of-line blocking and joins all work before restart.
		started := map[string]bool{}
		failed := make(chan error, 1)
		discover := func() error {
			ids, err := s.Inbox.IDs(life)
			if err != nil {
				return err
			}
			for _, id := range ids {
				if started[id] {
					continue
				}
				started[id] = true
				wg.Add(1)
				go func() {
					defer wg.Done()
					ticker := time.NewTicker(s.Interval)
					defer ticker.Stop()
					for {
						if err := s.step(life, id); err != nil {
							select {
							case failed <- err:
							default:
							}
							return
						}
						select {
						case <-life.Done():
							return
						case <-ticker.C:
						}
					}
				}()
			}
			return nil
		}
		if err := discover(); err != nil {
			return err
		}
		if ready != nil {
			ready()
		}
		ticker := time.NewTicker(s.Interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case err := <-failed:
				return err
			case <-ticker.C:
				if err := discover(); err != nil {
					return err
				}
			}
		}
	})
}
