package securekeys

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// Pending contains private material only in memory. Persist Handle() in a
// durable intent before Publish, then Close on every exit. Losing an unpublished
// Pending requires an explicit abort; it must not regenerate the intended key.
type Pending struct {
	mu      sync.Mutex
	store   *Store
	handle  Handle
	encoded []byte
}

func (s *Store) PrepareSigning() (*Pending, error)  { return s.prepare(Version) }
func (s *Store) PrepareEnvelope() (*Pending, error) { return s.prepare(EnvelopeVersion) }

func (s *Store) prepare(version string) (*Pending, error) {
	if s == nil || s.backend == nil {
		return nil, ErrUnavailable
	}
	seed := make([]byte, ed25519.SeedSize)
	defer clear(seed)
	if _, err := rand.Read(seed); err != nil {
		return nil, err
	}
	id := make([]byte, 16)
	if _, err := rand.Read(id); err != nil {
		return nil, err
	}
	r := record{Version: version, Scope: s.scope, ID: hex.EncodeToString(id), Seed: seed}
	public, err := publicKey(version, seed)
	if err != nil {
		return nil, err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	return &Pending{store: s, handle: Handle{Version: version, Backend: s.name, Scope: s.scope, ID: r.ID, PublicKey: public}, encoded: b}, nil
}

func (p *Pending) Handle() Handle {
	p.mu.Lock()
	defer p.mu.Unlock()
	h := p.handle
	h.PublicKey = bytes.Clone(h.PublicKey)
	return h
}

// Publish can be retried on the same live Pending after uncertain publication.
// An existing record must match the exact intended key, never just its ID.
func (p *Pending) Publish() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.encoded) == 0 {
		return fmt.Errorf("pending key closed")
	}
	err := p.store.backend.create(p.handle.ID, p.encoded)
	if errors.Is(err, ErrExists) {
		return p.store.Inspect(p.handle)
	}
	return err
}

func (p *Pending) Close() { p.mu.Lock(); defer p.mu.Unlock(); clear(p.encoded); p.encoded = nil }

// Inspect validates existence, backend, scope, profile and key equality without
// returning private bytes. Missing/corrupt keys never get recreated.
func (s *Store) Inspect(h Handle) error {
	b, seed, err := s.loadMaterial(h)
	clear(b)
	clear(seed)
	return err
}

func (s *Store) Backend() string { return s.name }
func (s *Store) Scope() string   { return s.scope }
