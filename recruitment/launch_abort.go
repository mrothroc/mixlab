package recruitment

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
)

func (s *Launch) compensate(ctx context.Context, p LaunchPorts) error {
	old, r, err := s.load()
	if err != nil {
		return err
	}
	if r.Phase == "succeeded" || r.Phase == "aborted" {
		return nil
	}
	if err := s.phase(&old, &r, "aborting"); err != nil {
		return err
	}
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		complete := true
		var failures []error
		for i := range r.Nodes {
			n := &r.Nodes[i]
			if !n.Touched || n.Released {
				continue
			}
			if err := ctx.Err(); err != nil {
				return errors.Join(fmt.Errorf("launch cleanup incomplete; retry abort using retained journal"), err, errors.Join(failures...))
			}
			if !n.Fenced {
				// Fence the original request even when its response was lost. This
				// cannot allocate another lease or race a delayed reserve request.
				f, err := p.AbortReservation(ctx, n.Candidate, n.Reserve)
				if err != nil {
					complete = false
					failures = append(failures, err)
					continue
				}
				b, _ := json.Marshal(n.Reserve)
				if f.Node != n.Candidate.Capabilities.Node || f.Controller != r.Plan.Controller || f.RequestHash != nodejob.Hash(b) || f.Lease != nil && (!leaseMatches(r.Plan, *n, *f.Lease) || (n.Lease != nil && f.Lease.ID != n.Lease.ID)) || f.Lease == nil && n.Lease != nil {
					return fmt.Errorf("ambiguous reservation identity; cleanup remains pending")
				}
				n.Lease = f.Lease
				n.Fenced = true
				n.Released = f.Lease == nil
				if err := s.save(&old, r); err != nil {
					return err
				}
			}
			if n.Released {
				continue
			}
			l, err := p.LeaseStatus(ctx, n.Candidate, n.Lease.ID)
			if err != nil {
				complete = false
				failures = append(failures, err)
				continue
			}
			if !leaseMatches(r.Plan, *n, l) || l.ID != n.Lease.ID {
				return fmt.Errorf("compensation lease identity mismatch")
			}
			if l.State == nodeagent.Released {
				n.Released = true
				if err := s.save(&old, r); err != nil {
					return err
				}
				continue
			}
			complete = false
			if l.State != nodeagent.Releasing {
				return fmt.Errorf("fenced reservation remains active")
			}
		}
		if complete {
			return s.phase(&old, &r, "aborted")
		}
		select {
		case <-ctx.Done():
			return errors.Join(fmt.Errorf("launch cleanup incomplete; retry abort using retained journal"), ctx.Err(), errors.Join(failures...))
		case <-ticker.C:
		}
	}
}
