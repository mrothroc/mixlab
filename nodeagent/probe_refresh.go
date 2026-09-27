package nodeagent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

// RefreshIdleProbe serializes the bounded approved child probe with reservation.
// Never initialize another GPU context while a lease owns the accelerator.
// Only volatile free memory and observation time may change;
// executable/device contracts still require administrator profile approval.
func (s *Store) RefreshIdleProbe(ctx context.Context, probe func(context.Context) (workerprobe.Report, error), clock func() time.Time) error {
	if probe == nil || clock == nil {
		return fmt.Errorf("bounded local probe and clock required")
	}
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, leases, err := s.load()
		if err != nil || leases.Active != "" {
			return err
		}
		old, p, err := s.loadProfile()
		if err != nil {
			return err
		}
		r, err := probe(ctx)
		if err != nil {
			return err
		}
		if err := r.Validate(); err != nil {
			return err
		}
		if !sameProbeContract(r, p.Probe) {
			return fmt.Errorf("approved worker/device contract changed")
		}
		p.Probe, p.ProbeObservedAt = r, clock().Unix()
		if err := p.Validate(); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		b, err := json.Marshal(p)
		if err != nil {
			return err
		}
		return s.path.CompareAndSwap(profileFile, old, b)
	})
}
