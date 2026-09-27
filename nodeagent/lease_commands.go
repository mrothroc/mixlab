package nodeagent

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func (s *Store) Reserve(ctx context.Context, actor trust.AuthenticatedPrincipal, q Reserve, now time.Time) (out Lease, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(q.IdempotencyKey) || !identifier(q.Run) || q.ExpectedNodeVersion == 0 || q.CapabilityGeneration == 0 {
		return out, fmt.Errorf("invalid reservation")
	}
	if err := ttl(q.TTLSeconds); err != nil {
		return out, err
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		for _, f := range r.ReservationFences {
			if f.Controller == actor.Principal && f.Key == q.IdempotencyKey {
				return fmt.Errorf("reservation permanently aborted")
			}
		}
		prior, ok, err := replay(r, actor.Principal, q.IdempotencyKey, "reserve", digest(q))
		if err != nil {
			return err
		}
		if ok {
			l := findLease(&r, prior.ID)
			if l == nil || l.State == Releasing || l.State == Released {
				return fmt.Errorf("reservation ended; submit a new request")
			}
			out = prior
			return nil
		}
		if r.Active != "" {
			return fmt.Errorf("node busy or awaiting terminal cleanup")
		}
		if q.ExpectedNodeVersion != r.NodeVersion || q.CapabilityGeneration != r.CapabilityGeneration {
			return fmt.Errorf("node capability/version changed")
		}
		if len(r.Leases) >= maxLeases {
			return fmt.Errorf("lease history capacity reached")
		}
		id, err := newID()
		if err != nil {
			return err
		}
		expires := min(now.Add(time.Duration(q.TTLSeconds)*time.Second).Unix(), actor.ExpiresAt.Unix())
		if expires <= now.Unix() {
			return fmt.Errorf("controller expires before reservation")
		}
		l := Lease{Format: LeaseVersion, ID: id, Node: s.node, Controller: actor.Principal, Run: q.Run, CapabilityGeneration: q.CapabilityGeneration, Version: 1, State: Reserved, Created: now.Unix(), Expires: expires, RenewBy: now.Unix() + (expires-now.Unix())/2}
		r.Leases = append(r.Leases, l)
		r.Active = id
		record(&r, actor.Principal, q.IdempotencyKey, "reserve", digest(q), l)
		if err := s.save(old, r); err != nil {
			return err
		}
		out = l
		return nil
	})
	return out, err
}

func (s *Store) RenewLease(ctx context.Context, actor trust.AuthenticatedPrincipal, q LeaseCommand, now time.Time) (Lease, error) {
	if err := ttl(q.TTLSeconds); err != nil {
		return Lease{}, err
	}
	return s.leaseCommand(ctx, actor, q, "renew", now)
}
func (s *Store) ReleaseLease(ctx context.Context, actor trust.AuthenticatedPrincipal, q LeaseCommand, now time.Time) (Lease, error) {
	if q.TTLSeconds != 0 {
		return Lease{}, fmt.Errorf("release cannot set a TTL")
	}
	return s.leaseCommand(ctx, actor, q, "release", now)
}

// LeaseStatus includes reconciled release, unlike a release command's immutable
// retry response. Controllers use it to confirm compensation actually finished.
func (s *Store) LeaseStatus(ctx context.Context, actor trust.AuthenticatedPrincipal, id string, now time.Time) (out Lease, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(id) {
		return out, fmt.Errorf("invalid lease ID")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		_, r, err := s.load()
		if err != nil {
			return err
		}
		for _, l := range r.Leases {
			if l.ID != id {
				continue
			}
			if l.Controller != actor.Principal {
				return fmt.Errorf("lease belongs to another controller")
			}
			out = l
			return nil
		}
		return fmt.Errorf("unknown lease")
	})
	return out, err
}
func (s *Store) leaseCommand(ctx context.Context, actor trust.AuthenticatedPrincipal, q LeaseCommand, action string, now time.Time) (out Lease, err error) {
	if err := authorize(actor, s.cluster, now); err != nil {
		return out, err
	}
	if !identifier(q.IdempotencyKey) || !identifier(q.Lease) || q.ExpectedVersion == 0 {
		return out, fmt.Errorf("invalid lease command")
	}
	err = s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		prior, ok, err := replay(r, actor.Principal, q.IdempotencyKey, action, digest(q))
		if err != nil {
			return err
		}
		if ok {
			out = prior
			return nil
		}
		for i := range r.Leases {
			l := &r.Leases[i]
			if l.ID != q.Lease {
				continue
			}
			if l.Controller != actor.Principal {
				return fmt.Errorf("lease belongs to another controller")
			}
			if l.Version != q.ExpectedVersion {
				return fmt.Errorf("lease version changed")
			}
			if action == "renew" {
				if l.State == Releasing || l.State == Released || now.Unix() >= l.Expires {
					return fmt.Errorf("lease cannot be renewed")
				}
				expires := min(now.Add(time.Duration(q.TTLSeconds)*time.Second).Unix(), actor.ExpiresAt.Unix())
				if expires <= l.Expires {
					return fmt.Errorf("renewal must extend lease expiry")
				}
				l.Expires = expires
				l.RenewBy = now.Unix() + (expires-now.Unix())/2
			} else {
				if l.State == Released {
					return fmt.Errorf("lease already released; retry original command")
				}
				if l.State == Releasing {
					record(&r, actor.Principal, q.IdempotencyKey, action, digest(q), *l)
					if err := s.save(old, r); err != nil {
						return err
					}
					out = *l
					return nil
				}
				l.State = Releasing
				l.CleanupReason = "released"
			}
			l.Version++
			record(&r, actor.Principal, q.IdempotencyKey, action, digest(q), *l)
			if err := s.save(old, r); err != nil {
				return err
			}
			out = *l
			return nil
		}
		return fmt.Errorf("unknown lease")
	})
	return out, err
}

// Expire records cleanup intent only. Even a passed deadline cannot advertise
// the accelerator as free while worker liveness remains uncertain.
func (s *Store) Expire(ctx context.Context, now time.Time) error {
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		for i := range r.Leases {
			l := &r.Leases[i]
			if l.ID == r.Active && l.State != Releasing && now.Unix() >= l.Expires {
				l.State = Releasing
				l.CleanupReason = "expired"
				l.Version++
				r.NodeVersion++
				return s.save(old, r)
			}
		}
		return nil
	})
}

// ReleaseUnprepared reconciles a reservation that never bound a job. A child
// can only be approved after that binding, so no hosting effect is possible.
// Prepared/running jobs require the separate terminal-outcome reconciliation.
func (s *Store) ReleaseUnprepared(ctx context.Context, id string) error {
	return s.path.WithProcessLock(ctx, nodeLock, func() error {
		old, r, err := s.load()
		if err != nil {
			return err
		}
		for i := range r.Leases {
			l := &r.Leases[i]
			if l.ID != id {
				continue
			}
			if l.Job != "" {
				return fmt.Errorf("bound job requires confirmed terminal cleanup")
			}
			if l.State == Released {
				return nil
			}
			if l.State != Releasing {
				return fmt.Errorf("release intent required")
			}
			l.State = Released
			l.Version++
			r.Active = ""
			r.NodeVersion++
			return s.save(old, r)
		}
		return fmt.Errorf("unknown lease")
	})
}
