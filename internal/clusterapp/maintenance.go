package clusterapp

import (
	"context"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"errors"
	"time"

	"github.com/mrothroc/mixlab/principalrenewal"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/principal"
)

func PrincipalScheduler(path statehome.Path, p *principal.Store, renew func(context.Context, time.Time) error) (*principalrenewal.Scheduler, error) {
	return principalrenewal.New(path, principalrenewal.Ports{
		Identity: func(now time.Time) (principalrenewal.Identity, error) {
			s, err := p.View(now)
			if err != nil {
				return principalrenewal.Identity{}, err
			}
			c, err := x509.ParseCertificate(s.Chain[0])
			if err != nil {
				return principalrenewal.Identity{}, err
			}
			d := sha256.Sum256(s.Chain[0])
			return principalrenewal.Identity{Principal: s.Principal, Certificate: hex.EncodeToString(d[:]), Issued: c.NotBefore, Expires: c.NotAfter}, nil
		}, Renew: renew,
	})
}

func authorityMaintenance(ctx context.Context, a *Authority, s *principalrenewal.Scheduler, clock func() time.Time, event func(error)) (result error) {
	defer func() {
		if ctx.Err() != nil && errors.Is(result, ctx.Err()) {
			result = nil
		}
	}()
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for {
		if _, err := a.Current(ctx, clock()); err != nil {
			return err
		}
		state, err := s.Tick(ctx, clock())
		if err != nil {
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
