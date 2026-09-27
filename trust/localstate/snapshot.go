// Package localstate persists cluster trust without owning application journals.
package localstate

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

const filename = "trust-state.json"
const maxStateBytes = 2 * trust.MaxTrustBytes
const version = "mixlab_local_trust_v1"

type record struct {
	Version     string          `json:"version"`
	Root        []byte          `json:"root"`
	Fingerprint string          `json:"fingerprint"`
	Snapshot    json.RawMessage `json:"snapshot"`
}

// Store holds a pin supplied by the owning principal/authority context. It
// cannot infer or replace a cluster identity from a snapshot or routing hint.
type Store struct {
	path        statehome.Path
	fingerprint string
}

func Open(path statehome.Path, fingerprint string) (*Store, error) {
	b, err := hex.DecodeString(fingerprint)
	if err != nil || len(b) != 32 || hex.EncodeToString(b) != fingerprint {
		return nil, fmt.Errorf("expected root fingerprint required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Store{path: path, fingerprint: fingerprint}, nil
}

func (s *Store) Initialize(a trust.Anchor, snapshot trust.SignedSnapshot, now time.Time) error {
	if a.Fingerprint() != s.fingerprint {
		return fmt.Errorf("root differs from configured trust")
	}
	v, err := trust.VerifySnapshot(a, snapshot, now)
	if err != nil {
		return err
	}
	b, err := encode(a, v)
	if err != nil {
		return err
	}
	return s.path.CompareAndSwap(filename, nil, b)
}

func encode(a trust.Anchor, v trust.VerifiedSnapshot) ([]byte, error) {
	snapshot, err := v.Bytes()
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(record{Version: version, Root: a.DER(), Fingerprint: a.Fingerprint(), Snapshot: snapshot})
	if len(b) > maxStateBytes {
		return nil, fmt.Errorf("trust state too large")
	}
	return b, err
}

func (s *Store) load(now time.Time) ([]byte, trust.Anchor, trust.VerifiedSnapshot, error) {
	var a trust.Anchor
	var v trust.VerifiedSnapshot
	b, err := s.path.ReadFileLimit(filename, maxStateBytes)
	if err != nil {
		return nil, a, v, err
	}
	var r record
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, a, v, fmt.Errorf("invalid local trust state")
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) || r.Version != version || r.Fingerprint != s.fingerprint {
		return nil, a, v, fmt.Errorf("local trust state version, encoding, or root mismatch")
	}
	a, err = trust.PinRoot(r.Root, s.fingerprint, now)
	if err != nil {
		return nil, a, v, err
	}
	v, err = trust.RestoreSnapshot(a, r.Snapshot, now)
	return b, a, v, err
}

// Load allows stale persisted snapshots only for subsequent refresh. Trust's
// signature-acceptance methods still enforce freshness before any operation.
func (s *Store) Load(now time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
	_, a, v, err := s.load(now)
	return a, v, err
}

// Advance commits signed bytes and generation together. A concurrent writer
// produces statehome.ErrConflict; callers must reload and revalidate, never
// retry an unconditional write. Missing/corrupt state does not reinitialize.
func (s *Store) Advance(next trust.SignedSnapshot, now time.Time) error {
	old, a, v, err := s.load(now)
	if err != nil {
		return err
	}
	n, err := v.Advance(next, now)
	if err != nil {
		return err
	}
	b, err := encode(a, n)
	if err != nil {
		return err
	}
	return s.path.CompareAndSwap(filename, old, b)
}
