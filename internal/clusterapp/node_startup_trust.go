package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/principal"
)

// RefreshNodeStartup lets an offline node recover stale public trust without
// renewing credentials or becoming recruitable first. A still-current cached
// snapshot tolerates an authority outage; expiry/revocation never does.
func RefreshNodeStartup(ctx context.Context, p *principal.Store, clock func() time.Time) error {
	if p == nil || clock == nil {
		return fmt.Errorf("node principal and clock required")
	}
	r, _, err := p.SnapshotReceiver(clock())
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	var failures error
	for _, endpoint := range r.Snapshot.Payload.Endpoints.Payload.URLs {
		err := RefreshRemoteTrust(ctx, p, endpoint, clock)
		if err == nil {
			_, _, err = p.Active(clock())
			return err
		}
		failures = errors.Join(failures, err)
		if ctx.Err() != nil {
			return errors.Join(failures, ctx.Err())
		}
	}
	_, _, err = p.Active(clock())
	if err == nil {
		return nil
	}
	return errors.Join(err, failures)
}
