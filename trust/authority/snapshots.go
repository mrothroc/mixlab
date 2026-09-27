// Package authority owns durable trust-snapshot publication and revocation.
// Only trusted local composition can open it with the protected snapshot signer.
// Network adapters must not turn possession of a TLS identity into admin access.
package authority

import (
	"bytes"
	"context"
	"crypto"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const filename = "authority-snapshots.json"
const lockname = "authority-snapshots.lock"
const version = "mixlab_authority_snapshots_v1"
const maxBytes = 2 * trust.MaxTrustBytes

type journal interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
	WithProcessLock(context.Context, string, func() error) error
}

type Store struct {
	store             journal
	anchor            trust.Anchor
	signerCertificate []byte
	signer            crypto.Signer
}

type record struct {
	Version     string               `json:"version"`
	Cluster     string               `json:"cluster"`
	Fingerprint string               `json:"fingerprint"`
	Snapshot    trust.SignedSnapshot `json:"snapshot"`
}

func configured(path statehome.Path, a trust.Anchor, signerDER []byte, signer crypto.Signer, now time.Time) (*Store, error) {
	if path.Kind() != statehome.Authority || signer == nil {
		return nil, fmt.Errorf("snapshot authority requires authority state and protected signer")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	c, _, err := certificates.Verify(a, signerDER, nil, certificates.SnapshotSigner, now)
	if err != nil {
		return nil, err
	}
	if err := certificates.MatchSigner(c, signer); err != nil {
		return nil, err
	}
	return &Store{store: path, anchor: a, signerCertificate: bytes.Clone(signerDER), signer: signer}, nil
}

// Initialize is explicit first publication. Never call it as a missing-state
// fallback: the journal contains revocation history that cannot be discarded.
func Initialize(ctx context.Context, path statehome.Path, a trust.Anchor, signerDER []byte, signer crypto.Signer, initial trust.SignedSnapshot, now time.Time) (*Store, error) {
	s, err := configured(path, a, signerDER, signer, now)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(initial.SignerCertificate, signerDER) {
		return nil, fmt.Errorf("initial snapshot signer mismatch")
	}
	initialBytes, err := json.Marshal(initial)
	if err != nil {
		return nil, err
	}
	if _, err := trust.RestoreSnapshot(a, initialBytes, now); err != nil {
		return nil, err
	}
	err = path.WithProcessLock(ctx, lockname, func() error { return s.save(nil, initial) })
	if err != nil {
		return nil, err
	}
	return s, nil
}

func Open(path statehome.Path, a trust.Anchor, signerDER []byte, signer crypto.Signer, now time.Time) (*Store, error) {
	s, err := configured(path, a, signerDER, signer, now)
	if err != nil {
		return nil, err
	}
	_, _, _, err = s.load(now)
	if err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) load(now time.Time) ([]byte, trust.SignedSnapshot, trust.VerifiedSnapshot, error) {
	var r record
	var v trust.VerifiedSnapshot
	b, err := s.store.ReadFileLimit(filename, maxBytes)
	if err != nil {
		return nil, r.Snapshot, v, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r.Snapshot, v, fmt.Errorf("invalid snapshot authority journal")
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) || r.Version != version || r.Cluster != s.anchor.Cluster() || r.Fingerprint != s.anchor.Fingerprint() || !bytes.Equal(r.Snapshot.SignerCertificate, s.signerCertificate) {
		return nil, r.Snapshot, v, fmt.Errorf("snapshot authority context mismatch")
	}
	body, err := json.Marshal(r.Snapshot)
	if err != nil {
		return nil, r.Snapshot, v, err
	}
	v, err = trust.RestoreSnapshot(s.anchor, body, now)
	return b, r.Snapshot, v, err
}

func (s *Store) save(old []byte, snapshot trust.SignedSnapshot) error {
	b, err := json.Marshal(record{version, s.anchor.Cluster(), s.anchor.Fingerprint(), snapshot})
	if err != nil {
		return err
	}
	if len(b) > maxBytes {
		return fmt.Errorf("snapshot authority journal too large")
	}
	return s.store.CompareAndSwap(filename, old, b)
}

// Load permits stale history for subsequent Refresh only. Consumers still use
// trust.VerifySnapshot or principal authentication before new operations.
func (s *Store) Load(now time.Time) (trust.SignedSnapshot, error) {
	_, snapshot, _, err := s.load(now)
	return snapshot, err
}

// Refresh publishes a new generation when less than one third of the snapshot
// lifetime remains. It retains every revocation and never extends root-signed
// endpoint or issuer lifetimes. Expired signing material requires explicit rekey.
func (s *Store) Refresh(ctx context.Context, now time.Time) (trust.SignedSnapshot, error) {
	var result trust.SignedSnapshot
	err := s.store.WithProcessLock(ctx, lockname, func() error {
		old, current, v, err := s.load(now)
		if err != nil {
			return err
		}
		if _, err := trust.VerifySnapshot(s.anchor, current, now); err == nil && current.Payload.ExpiresAt > now.Add(trust.SnapshotLifetime/3).Unix() {
			result = current
			return nil
		}
		result, err = s.publish(old, current, v, now)
		return err
	})
	if err != nil {
		return trust.SignedSnapshot{}, err
	}
	return result, nil
}

func (s *Store) publish(old []byte, current trust.SignedSnapshot, v trust.VerifiedSnapshot, now time.Time) (trust.SignedSnapshot, error) {
	if current.Payload.Generation == ^uint64(0) {
		return trust.SignedSnapshot{}, fmt.Errorf("trust generation exhausted")
	}
	current.Payload.Generation++
	current.Payload.IssuedAt, current.Payload.ExpiresAt = now.Unix(), now.Add(trust.SnapshotLifetime).Unix()
	next, err := trust.SignSnapshot(s.anchor, current.Payload, s.signerCertificate, s.signer, now)
	if err != nil {
		return trust.SignedSnapshot{}, err
	}
	if _, err := v.Advance(next, now); err != nil {
		return trust.SignedSnapshot{}, err
	}
	if err := s.save(old, next); err != nil {
		return trust.SignedSnapshot{}, err
	}
	return next, nil
}
