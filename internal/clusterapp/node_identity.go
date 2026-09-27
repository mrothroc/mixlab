package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/principalrenewal"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

func NodePrincipalPolicy(p *principal.Store, clock func() time.Time) (*managedtls.Policy, error) {
	if p == nil || clock == nil {
		return nil, fmt.Errorf("node principal and clock required")
	}
	s, _, err := p.Active(clock())
	if err != nil {
		return nil, err
	}
	if s.Role != trust.Node {
		return nil, fmt.Errorf("agent requires a node principal")
	}
	return managedPrincipalPolicy(p, trust.Controller, clock)
}

func NodeCurrentTrust(p *principal.Store) func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
	return func(ctx context.Context, now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		if p == nil {
			return trust.Anchor{}, trust.VerifiedSnapshot{}, fmt.Errorf("node principal required")
		}
		if err := ctx.Err(); err != nil {
			return trust.Anchor{}, trust.VerifiedSnapshot{}, err
		}
		s, _, err := p.Active(now)
		if err != nil {
			return trust.Anchor{}, trust.VerifiedSnapshot{}, err
		}
		if s.Role != trust.Node {
			return trust.Anchor{}, trust.VerifiedSnapshot{}, fmt.Errorf("node principal required")
		}
		a, err := trust.PinRoot(s.Root, s.Fingerprint, now)
		if err != nil {
			return trust.Anchor{}, trust.VerifiedSnapshot{}, err
		}
		v, err := trust.VerifySnapshot(a, s.Snapshot, now)
		return a, v, err
	}
}

// MaintainNodeTrust tries only root-authorized endpoints. Temporary authority
// outages retain the installed snapshot, which still expires normally; neither
// TLS nor execution may keep using stale trust while refresh is retried.
func MaintainNodeTrust(ctx context.Context, path statehome.Path, p *principal.Store, clock func() time.Time, event func(error)) error {
	return maintainPrincipalTrust(ctx, path, p, trust.Node, clock, event)
}

// MaintainControllerTrust uses the same signed-endpoint renewal policy as a
// node, without giving recruitment access to authority keys.
func MaintainControllerTrust(ctx context.Context, path statehome.Path, p *principal.Store, clock func() time.Time, event func(error)) error {
	return maintainPrincipalTrust(ctx, path, p, trust.Controller, clock, event)
}

func maintainPrincipalTrust(ctx context.Context, path statehome.Path, p *principal.Store, role trust.Role, clock func() time.Time, event func(error)) error {
	if p == nil || clock == nil {
		return fmt.Errorf("principal and clock required")
	}
	refresh := func(ctx context.Context, renew bool) error {
		s, err := p.View(clock())
		if err != nil {
			return err
		}
		if s.Role != role {
			return fmt.Errorf("%s principal required", role)
		}
		var last error
		for _, endpoint := range s.Snapshot.Payload.Endpoints.Payload.URLs {
			attempt, done := context.WithTimeout(ctx, 20*time.Second)
			if renew {
				last = RenewRemotePrincipal(attempt, p, endpoint, clock)
			} else {
				last = RefreshRemoteTrust(attempt, p, endpoint, clock)
			}
			done()
			if last == nil || ctx.Err() != nil {
				return last
			}
		}
		return last
	}
	scheduler, err := PrincipalScheduler(path, p, func(ctx context.Context, _ time.Time) error { return refresh(ctx, true) })
	if err != nil {
		return err
	}
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		if err := refresh(ctx, false); err != nil && event != nil && ctx.Err() == nil {
			event(fmt.Errorf("%s trust refresh: %w", role, err))
		}
		if ctx.Err() != nil {
			return nil
		}
		state, err := scheduler.Tick(ctx, clock())
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			if errors.Is(err, principalrenewal.ErrReenrollmentRequired) || state.Outcome != "failed" {
				return err
			}
			if event != nil {
				event(err)
			}
		}
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
		}
	}
}
