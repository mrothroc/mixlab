package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/principal"
)

func refreshPrincipalTrust(ctx context.Context, p *principal.Store, clock func() time.Time) error {
	if p == nil || clock == nil {
		return fmt.Errorf("principal and clock required")
	}
	return retryTrustRefresh(ctx, func(call context.Context, attempt int) error {
		return refreshPrincipalTrustAttempt(call, p, clock, attempt)
	})
}

func refreshPrincipalTrustAttempt(ctx context.Context, p *principal.Store, clock func() time.Time, attempt int) error {
	// Reload signed destinations after every wait; peer evidence never
	// supplies endpoints and endpoint removal is effective immediately.
	s, err := p.View(clock())
	if err != nil {
		return err
	}
	endpoints := s.Snapshot.Payload.Endpoints.Payload.URLs
	if len(endpoints) == 0 {
		return fmt.Errorf("signed authority endpoint missing")
	}
	return refreshRemoteTrustWithTimeout(ctx, p, endpoints[attempt%len(endpoints)], clock, remoteTrustRefreshTimeout)
}

// retryTrustRefresh retries only read-only snapshot fetches. Unlike startup's
// explicit cached-trust fallback, exhausting this budget always returns error.
func retryTrustRefresh(parent context.Context, refresh func(context.Context, int) error) error {
	return retryTrustRefreshWithCache(parent, refresh, nil)
}

func retryTrustRefreshWithCache(parent context.Context, refresh func(context.Context, int) error, active func() error) error {
	ctx, cancel := context.WithTimeout(parent, remoteTrustRefreshBudget)
	defer cancel()
	delay := 250 * time.Millisecond
	var last, cacheErr error
	for attempt := 0; ctx.Err() == nil; attempt++ {
		call, stop := context.WithTimeout(ctx, remoteTrustRefreshTimeout)
		err := refresh(call, attempt)
		if err == nil {
			err = call.Err()
		}
		stop()
		if ctx.Err() != nil {
			return errors.Join(err, cacheErr, ctx.Err())
		}
		if err == nil {
			return ctx.Err()
		}
		if !transientControllerTrust(err) {
			return err
		}
		last = err
		if active != nil {
			// Startup may proceed after one transient failure, but only with
			// trust still valid now, after the potentially slow attempt.
			cacheErr = active()
			if ctx.Err() != nil {
				return errors.Join(err, cacheErr, ctx.Err())
			}
			if cacheErr == nil {
				return nil
			}
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
		delay = min(2*delay, 2*time.Second)
	}
	return errors.Join(last, cacheErr, ctx.Err())
}

func refreshStartupTrust(ctx context.Context, p *principal.Store, clock func() time.Time) error {
	if p == nil || clock == nil {
		return fmt.Errorf("principal and clock required")
	}
	active := func() error {
		_, _, err := p.Active(clock())
		return err
	}
	err := retryTrustRefreshWithCache(ctx, func(call context.Context, attempt int) error {
		return refreshPrincipalTrustAttempt(call, p, clock, attempt)
	}, active)
	if err != nil {
		return err
	}
	// Validate successful refreshes too; installing a signed snapshot is not
	// itself proof that the local identity remains authorized.
	activeErr := active()
	if ctx.Err() != nil {
		return errors.Join(activeErr, ctx.Err())
	}
	return activeErr
}
