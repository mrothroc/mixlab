package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

type reservationFence struct{ Controller, Key, Digest string }

type ReservationAbort struct {
	Node        string `json:"node"`
	Controller  string `json:"controller"`
	RequestHash string `json:"request_hash"`
	Lease       *Lease `json:"lease"`
}

// AbortReservation serializes with Reserve and records a permanent tombstone.
// A nil lease proves no request with this controller/key can create a lease
// later, including an original request delayed across the network.
func (s *Store) AbortReservation(ctx context.Context, actor trust.AuthenticatedPrincipal, q Reserve, now time.Time) (out ReservationAbort, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(q.IdempotencyKey) || !identifier(q.Run) || q.ExpectedNodeVersion == 0 || q.CapabilityGeneration == 0 {
		return out, fmt.Errorf("exact reservation required")
	}
	if err := ttl(q.TTLSeconds); err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		hash := digest(q)
		out = ReservationAbort{Node: s.node, Controller: actor.Principal, RequestHash: hash}
		found := false
		for _, f := range r.ReservationFences {
			if f.Controller == actor.Principal && f.Key == q.IdempotencyKey {
				if f.Digest != hash {
					return fmt.Errorf("abort reservation payload changed")
				}
				found = true
			}
		}
		for _, c := range r.Receipts {
			if c.Controller != actor.Principal || c.Key != q.IdempotencyKey {
				continue
			}
			if c.Action != "reserve" || c.Digest != hash {
				return fmt.Errorf("abort reservation identity mismatch")
			}
			l := findLease(&r, c.Result.ID)
			if l == nil {
				return fmt.Errorf("reservation history missing")
			}
			if l.State != Releasing && l.State != Released {
				l.State = Releasing
				l.CleanupReason = "released"
				l.Version++
				r.NodeVersion++
			}
			copy := *l
			out.Lease = &copy
		}
		// An existing immutable reserve receipt plus its terminal lease is
		// already a fence. Never let tombstone capacity prevent its release.
		if !found && out.Lease == nil {
			if len(r.ReservationFences) >= maxReceipts {
				return fmt.Errorf("reservation fence history capacity reached")
			}
			r.ReservationFences = append(r.ReservationFences, reservationFence{actor.Principal, q.IdempotencyKey, hash})
			r.NodeVersion++
		}
		return s.save(old, r)
	})
	return out, err
}
