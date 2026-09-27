// Package principal owns installed principal credentials and their atomic
// refresh/renewal publication. It does not issue certificates or open listeners.
package principal

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
)

const Version = "mixlab_principal_state_v1"
const bootstrapVersion = "mixlab_cluster_bootstrap_v1"
const filename = "principal.json"
const lockname = "principal.lock"
const maxBytes = 2 * trust.MaxTrustBytes

type State struct {
	Version     string               `json:"version"`
	Cluster     string               `json:"cluster"`
	Role        trust.Role           `json:"role"`
	Principal   string               `json:"principal"`
	Key         securekeys.Handle    `json:"key"`
	Root        []byte               `json:"root"`
	Fingerprint string               `json:"fingerprint"`
	Chain       [][]byte             `json:"chain"`
	Snapshot    trust.SignedSnapshot `json:"snapshot"`
	EnvelopeKey *securekeys.Handle   `json:"envelope_key,omitempty"`
}

type Store struct {
	path           statehome.Path
	keys           *securekeys.Store
	pin, principal string
	role           trust.Role
}

func decode(b []byte) (State, error) {
	var s State
	if len(b) > maxBytes {
		return s, fmt.Errorf("principal state too large")
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return s, fmt.Errorf("invalid principal state")
	}
	again, err := json.Marshal(s)
	if err != nil || !bytes.Equal(b, again) || (s.Version != Version && s.Version != bootstrapVersion) || !certificates.ValidID(s.Cluster) || !certificates.ValidID(s.Principal) {
		return State{}, fmt.Errorf("noncanonical or invalid principal state")
	}
	switch s.Role {
	case trust.Authority, trust.Controller, trust.Coordinator, trust.Node:
	default:
		return State{}, fmt.Errorf("invalid installed principal role")
	}
	return s, nil
}

// Open never initializes missing state or chooses another key-store backend.
// Stale trust and an expired leaf may be inspected; Active still fails closed.
func Open(p statehome.Path, now time.Time) (*Store, error) {
	if p.Kind() != statehome.Principal {
		return nil, fmt.Errorf("principal context required")
	}
	b, err := p.ReadFileLimit(filename, maxBytes)
	if err != nil {
		return nil, err
	}
	r, err := decode(b)
	if err != nil {
		return nil, err
	}
	keys, err := securekeys.OpenSelected(r.Key.Backend, p, r.Key.Scope)
	if err != nil {
		return nil, err
	}
	s := &Store{path: p, keys: keys, pin: r.Fingerprint, principal: r.Principal, role: r.Role}
	if _, _, _, err := s.load(now); err != nil {
		_ = keys.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error { return s.keys.Close() }

func (s *Store) validate(r State, now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
	var zero trust.VerifiedSnapshot
	a, err := trust.PinRoot(r.Root, s.pin, now)
	if err != nil {
		return a, zero, err
	}
	if r.Fingerprint != s.pin || r.Cluster != a.Cluster() || r.Principal != s.principal || r.Role != s.role {
		return a, zero, fmt.Errorf("installed principal identity changed")
	}
	b, err := json.Marshal(r.Snapshot)
	if err != nil {
		return a, zero, err
	}
	v, err := trust.RestoreSnapshot(a, b, now)
	if err != nil {
		return a, v, err
	}
	if len(r.Chain) != 3 || !bytes.Equal(r.Chain[2], a.DER()) {
		return a, v, fmt.Errorf("principal chain mismatch")
	}
	leaf, id, err := certificates.Parse(r.Chain[0])
	if err != nil {
		return a, v, err
	}
	// Validate the original signed credential at its issuance time for storage
	// inspection. This is not authentication or permission to renew after expiry.
	if _, _, err = certificates.Verify(a, r.Chain[0], r.Chain[1], certificates.Principal, leaf.NotBefore.Add(time.Second)); err != nil {
		return a, v, err
	}
	if id.Principal != r.Principal || id.Role != r.Role || !bytes.Equal(leaf.PublicKey.(ed25519.PublicKey), r.Key.PublicKey) {
		return a, v, fmt.Errorf("principal certificate/key binding mismatch")
	}
	m, err := keylifecycle.Open(s.path, s.keys, string(r.Role))
	if err != nil {
		return a, v, err
	}
	if err := activeHandle(m, keylifecycle.Principal, r.Key); err != nil {
		return a, v, err
	}
	if _, err := s.keys.Signer(r.Key); err != nil {
		return a, v, err
	}
	if r.Role == trust.Node {
		if r.EnvelopeKey == nil || !bytes.Equal(id.EnvelopeKey, r.EnvelopeKey.PublicKey) {
			return a, v, fmt.Errorf("node envelope handle mismatch")
		}
		if err := activeHandle(m, keylifecycle.NodeEnvelope, *r.EnvelopeKey); err != nil {
			return a, v, err
		}
		if _, err := s.keys.EnvelopeOpener(*r.EnvelopeKey); err != nil {
			return a, v, err
		}
	} else if r.EnvelopeKey != nil || len(id.EnvelopeKey) != 0 {
		return a, v, fmt.Errorf("envelope key on non-node principal")
	}
	return a, v, nil
}

func activeHandle(m *keylifecycle.Manager, slot keylifecycle.Slot, h securekeys.Handle) error {
	r, err := m.View(slot)
	if err != nil {
		return err
	}
	a, _ := json.Marshal(r.Active)
	b, _ := json.Marshal(h)
	if r.Stage != "active" || r.Active == nil || !bytes.Equal(a, b) {
		return fmt.Errorf("installed credential does not match active key lifecycle")
	}
	return nil
}

func (s *Store) load(now time.Time) ([]byte, State, trust.VerifiedSnapshot, error) {
	b, err := s.path.ReadFileLimit(filename, maxBytes)
	if err != nil {
		return nil, State{}, trust.VerifiedSnapshot{}, err
	}
	r, err := decode(b)
	if err != nil {
		return nil, r, trust.VerifiedSnapshot{}, err
	}
	_, v, err := s.validate(r, now)
	return b, r, v, err
}

func (s *Store) View(now time.Time) (State, error) { _, r, _, err := s.load(now); return r, err }

func (s *Store) Active(now time.Time) (State, crypto.Signer, error) {
	_, r, v, err := s.load(now)
	if err != nil {
		return State{}, nil, err
	}
	a, err := trust.PinRoot(r.Root, s.pin, now)
	if err != nil {
		return State{}, nil, err
	}
	if _, err := trust.AuthenticatePrincipal(a, v, r.Chain, now); err != nil {
		return State{}, nil, err
	}
	k, err := s.keys.Signer(r.Key)
	return r, k, err
}

// Refresh advances signed trust or acknowledges the identical current signed
// value. It never rolls back, renews a certificate, or replaces identity/keys.
func (s *Store) Refresh(ctx context.Context, next trust.SignedSnapshot, now time.Time) error {
	return s.path.WithProcessLock(ctx, lockname, func() error {
		old, r, v, err := s.load(now)
		if err != nil {
			return err
		}
		a, err := trust.PinRoot(r.Root, r.Fingerprint, now)
		if err != nil {
			return err
		}
		if _, err := currentOrAdvance(a, v, next, now); err != nil {
			return err
		}
		previous, _ := json.Marshal(r.Snapshot)
		incoming, _ := json.Marshal(next)
		if bytes.Equal(previous, incoming) {
			return nil
		}
		r.Snapshot = next
		return s.save(old, r)
	})
}

func (s *Store) save(old []byte, r State) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if len(b) > maxBytes {
		return fmt.Errorf("principal state too large")
	}
	return s.path.CompareAndSwap(filename, old, b)
}

func currentOrAdvance(a trust.Anchor, v trust.VerifiedSnapshot, next trust.SignedSnapshot, now time.Time) (trust.VerifiedSnapshot, error) {
	previous, err := v.Bytes()
	if err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	b, err := json.Marshal(next)
	if err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	if bytes.Equal(previous, b) {
		return trust.VerifySnapshot(a, next, now)
	}
	return v.Advance(next, now)
}

// Renew publishes a same-key certificate only while the existing identity is
// still eligible. Expiry/revocation requires explicit new enrollment instead.
func (s *Store) Renew(ctx context.Context, chain [][]byte, snapshot trust.SignedSnapshot, now time.Time) error {
	return s.path.WithProcessLock(ctx, lockname, func() error {
		old, r, v, err := s.load(now)
		if err != nil {
			return err
		}
		a, err := trust.PinRoot(r.Root, s.pin, now)
		if err != nil {
			return err
		}
		next, err := currentOrAdvance(a, v, snapshot, now)
		if err != nil {
			return err
		}
		if _, err := trust.AuthenticatePrincipal(a, next, r.Chain, now); err != nil {
			return err
		}
		r.Chain, r.Snapshot = chain, snapshot
		if _, _, err := s.validate(r, now); err != nil {
			return err
		}
		if _, err := trust.AuthenticatePrincipal(a, next, chain, now); err != nil {
			return err
		}
		return s.save(old, r)
	})
}
