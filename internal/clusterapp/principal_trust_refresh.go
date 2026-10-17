package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/principal"
)

func refreshPrincipalTrust(ctx context.Context, p *principal.Store, clock func() time.Time) error {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	s, err := p.View(clock())
	if err != nil {
		return err
	}
	var failures error
	for _, endpoint := range s.Snapshot.Payload.Endpoints.Payload.URLs {
		if err = RefreshRemoteTrust(ctx, p, endpoint, clock); err == nil {
			return nil
		}
		failures = errors.Join(failures, err)
		if ctx.Err() != nil {
			return errors.Join(failures, ctx.Err())
		}
	}
	if failures == nil {
		return fmt.Errorf("signed authority endpoint missing")
	}
	return failures
}
