//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

type resourceProcessControl interface {
	suspend(context.Context) error
	resume(context.Context) error
}

// stop joins the only sampler. Physical cleanup remains owned by Supervisor.Run
// and precedes that join, including when filesystem sampling cannot be canceled.
func watchResources(parent context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler) func() error {
	return watchResourceMonitor(parent, cancel, p, dir, limits, sample, nil)
}

func watchLearnerResources(parent context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler) func() error {
	return watchResourceMonitor(parent, cancel, p, dir, limits, sample, p)
}

func watchResourceMonitor(parent context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler, control resourceProcessControl) func() error {
	if sample == nil {
		sample = sampleResources
	}
	ctx, stop := context.WithCancel(parent)
	done := make(chan error, 1)
	go func() {
		err := monitorResources(ctx, cancel, p, dir, limits, sample, control, time.Now)
		if err != nil {
			cancel()
			err = fmt.Errorf("worker resource monitor: %w", err)
		}
		done <- err
	}()
	return func() error {
		stop()
		return <-done
	}
}

type resourceObservation struct {
	usage    resourceUsage
	err      error
	probeErr error
}

func monitorResources(ctx context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler, control resourceProcessControl, now func() time.Time) error {
	var meter resourceMeter
	timeouts := 0
	paused := false
	lastHealthy := now()
	ticker := time.NewTicker(resourceInterval)
	defer ticker.Stop()
	freshness := time.NewTimer(resourceSampleGrace)
	defer freshness.Stop()
	freshC := freshness.C
	suspend := func() error {
		if paused || ctx.Err() != nil {
			return nil
		}
		if control == nil {
			// Plain workers retain their existing fail-closed timeout policy.
			return fmt.Errorf("no healthy resource sample within tolerance: %w", context.DeadlineExceeded)
		}
		if err := control.suspend(ctx); err != nil {
			return fmt.Errorf("suspend unmeasured worker: %w", err)
		}
		paused = true
		freshness.Stop()
		freshC = nil
		return nil
	}
	for {
		if ctx.Err() != nil {
			return nil
		}
		startedPaused := paused
		probe, stopProbe := context.WithTimeout(ctx, resourceSampleTimeout)
		result := make(chan resourceObservation, 1)
		go func() {
			u, err := sample(probe, p, dir)
			result <- resourceObservation{u, err, probe.Err()}
		}()
		// Only one sampler may exist. A stale observation can stop the group
		// while disk I/O is blocked, but neither launches another sampler nor
		// releases the PID ownership fence held by process sampling.
		var observation resourceObservation
		var controlErr error
	await:
		for {
			select {
			case observation = <-result:
				break await
			case <-ctx.Done():
				stopProbe()
				observation = <-result
				break await
			case <-freshC:
				if controlErr = suspend(); controlErr != nil {
					cancel()
					stopProbe()
					observation = <-result
					break await
				}
			}
		}
		stopProbe()
		err := observation.err
		// A valid process observation or disk lower bound remains evidence if
		// a subsequent probe times out. Enforce it before transient handling.
		if observedErr := meter.checkObservation(observation.usage, limits, err == nil); observedErr != nil {
			err = errors.Join(observedErr, err)
		}
		if err == nil {
			err = observation.probeErr
		}
		if ctx.Err() != nil {
			if resourceSampleInterrupted(err) {
				err = nil
			}
			return errors.Join(controlErr, err)
		}
		if err != nil {
			_, processTimeout := err.(*processSampleTimeoutError)
			if !processTimeout && (control == nil || !resourceSampleSlow(err, observation.probeErr)) {
				return err
			}
		}
		// A result and the freshness timer may become runnable together after
		// a scheduler stall. Do not let selecting the result skip suspension.
		if !paused && now().Sub(lastHealthy) >= resourceSampleGrace {
			if err := suspend(); err != nil {
				return err
			}
		}
		if err != nil {
			if timeouts < 2 {
				timeouts++
			}
			if timeouts >= 2 {
				if err := suspend(); err != nil {
					return err
				}
			}
		} else if !paused || startedPaused {
			// A sample begun before STOP cannot authorize CONT, even if it
			// completed successfully. Require a complete post-STOP observation.
			if paused {
				if err := control.resume(ctx); err != nil {
					return fmt.Errorf("resume measured worker: %w", err)
				}
				paused = false
			}
			timeouts = 0
			lastHealthy = now()
			freshness.Reset(resourceSampleGrace)
			freshC = freshness.C
		}
		select {
		case <-ctx.Done():
			return nil
		case <-freshC:
			if err := suspend(); err != nil {
				return err
			}
		case <-ticker.C:
		}
	}
}

func resourceSampleInterrupted(err error) bool {
	switch err.(type) {
	case *processSampleTimeoutError, *processSampleCanceledError:
		return true
	default:
		return err == context.Canceled || err == context.DeadlineExceeded
	}
}

func resourceSampleSlow(err, probeErr error) bool {
	// Never unwrap joined failures: a timeout plus a permission/parse error is
	// still fatal. Bare deadlines are transient only when this probe expired.
	if _, ok := err.(*processSampleTimeoutError); ok {
		return true
	}
	if probeErr != context.DeadlineExceeded {
		return false
	}
	if _, ok := err.(*processSampleCanceledError); ok {
		return errors.Is(err, context.DeadlineExceeded)
	}
	return err == context.DeadlineExceeded
}
