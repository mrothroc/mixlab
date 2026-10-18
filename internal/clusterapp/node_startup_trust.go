package clusterapp

import (
	"context"
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
	_, _, err := p.SnapshotReceiver(clock())
	if err != nil {
		return err
	}
	return refreshStartupTrust(ctx, p, clock)
}
