package authority

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

// Revoke is an administrator-local application port, not a TLS role grant.
// Retries with the same target/mode/reason preserve the first generation.
// Compromise escalation is allowed; reactivation or downgrading is not.
func (s *Store) Revoke(ctx context.Context, kind, id, mode, reason string, now time.Time) (trust.SignedSnapshot, error) {
	if (kind != "principal" && kind != "certificate") || !certificates.ValidID(id) || (mode != "prospective" && mode != "compromise") || reason == "" || len(reason) > 256 {
		return trust.SignedSnapshot{}, fmt.Errorf("invalid revocation")
	}
	for _, c := range reason {
		if c < 0x20 || c > 0x7e {
			return trust.SignedSnapshot{}, fmt.Errorf("invalid revocation reason")
		}
	}
	var result trust.SignedSnapshot
	err := s.store.WithProcessLock(ctx, lockname, func() error {
		old, current, v, err := s.load(now)
		if err != nil {
			return err
		}
		found := false
		for i := range current.Payload.Revocations {
			r := &current.Payload.Revocations[i]
			if r.Kind != kind || r.ID != id {
				continue
			}
			found = true
			if r.Mode == "compromise" && mode != "compromise" {
				return fmt.Errorf("revocation downgrade is forbidden")
			}
			if r.Mode == mode && r.Reason == reason {
				if _, err := trust.VerifySnapshot(s.anchor, current, now); err == nil {
					result = current
					return nil
				}
			}
			r.Mode, r.Reason = mode, reason
			break
		}
		if current.Payload.Generation == ^uint64(0) {
			return fmt.Errorf("trust generation exhausted")
		}
		if !found {
			current.Payload.Revocations = append(current.Payload.Revocations, trust.Revocation{Kind: kind, ID: id, Mode: mode, Reason: reason, FirstGeneration: current.Payload.Generation + 1})
		}
		sort.Slice(current.Payload.Revocations, func(i, j int) bool {
			a, b := current.Payload.Revocations[i], current.Payload.Revocations[j]
			return a.Kind+":"+a.ID < b.Kind+":"+b.ID
		})
		result, err = s.publish(old, current, v, now)
		return err
	})
	if err != nil {
		return trust.SignedSnapshot{}, err
	}
	return result, nil
}
