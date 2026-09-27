// Package securekeys stores private signing and envelope keys behind scoped handles.
// It does not issue identities, authorize signatures, or select trust policy.
package securekeys

import (
	"bytes"
	"crypto"
	"crypto/ecdh"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

const Version = "mixlab_signing_key_v1"
const EnvelopeVersion = "mixlab_envelope_key_v1"

var (
	ErrMissing     = errors.New("private key unavailable; explicit recovery required")
	ErrExists      = errors.New("private key already exists")
	ErrUnavailable = errors.New("key-store backend unavailable; explicit backend selection required")
)

// Handle is public state. Scope is a composition-owned context identifier
// (64 lowercase hex characters), not a filesystem path or authorization grant.
type Handle struct {
	Version   string `json:"version"`
	Backend   string `json:"backend"`
	Scope     string `json:"scope"`
	ID        string `json:"id"`
	PublicKey []byte `json:"public_key"`
}

type secretBackend interface {
	create(id string, value []byte) error
	read(id string) ([]byte, error)
	remove(id string, expected []byte) error
}

type Store struct {
	backend     secretBackend
	name, scope string
}

// Close releases native backend resources; file stores need no close action.
func (s *Store) Close() error {
	if s == nil {
		return nil
	}
	if c, ok := s.backend.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

func newStore(backend secretBackend, name, scope string) (*Store, error) {
	if !validHex(scope, 32) {
		return nil, fmt.Errorf("invalid key-store context scope")
	}
	return &Store{backend: backend, name: name, scope: scope}, nil
}

func validHex(s string, size int) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == size && hex.EncodeToString(b) == s
}

type record struct {
	Version string `json:"version"`
	Scope   string `json:"scope"`
	ID      string `json:"id"`
	Seed    []byte `json:"seed"`
}

// Generate returns the candidate handle even when publication reports an
// error: the storage commit may have succeeded before its durability check.
// The owning lifecycle must reconcile that exact handle, not blindly retry.
func (s *Store) Generate() (Handle, error) {
	return s.generate(Version)
}

func (s *Store) generate(version string) (Handle, error) {
	p, err := s.prepare(version)
	if err != nil {
		return Handle{}, err
	}
	defer p.Close()
	h := p.Handle()
	return h, p.Publish()
}

func (s *Store) validate(h Handle) error {
	if s == nil || s.backend == nil {
		return ErrUnavailable
	}
	if (h.Version != Version && h.Version != EnvelopeVersion) || h.Backend != s.name || h.Scope != s.scope || !validHex(h.ID, 16) || len(h.PublicKey) != ed25519.PublicKeySize {
		return fmt.Errorf("key handle does not match backend/context/profile")
	}
	return nil
}

func (s *Store) load(h Handle) ([]byte, ed25519.PrivateKey, error) {
	if h.Version != Version {
		return nil, nil, fmt.Errorf("not a signing-key handle")
	}
	b, seed, err := s.loadMaterial(h)
	if err != nil {
		return nil, nil, err
	}
	defer clear(seed)
	return b, ed25519.NewKeyFromSeed(seed), nil
}

func publicKey(version string, seed []byte) ([]byte, error) {
	switch version {
	case Version:
		k := ed25519.NewKeyFromSeed(seed)
		defer clear(k)
		return bytes.Clone(k[ed25519.SeedSize:]), nil
	case EnvelopeVersion:
		k, err := ecdh.X25519().NewPrivateKey(seed)
		if err != nil {
			return nil, err
		}
		return k.PublicKey().Bytes(), nil
	default:
		return nil, fmt.Errorf("unknown key profile")
	}
}

func (s *Store) loadMaterial(h Handle) ([]byte, []byte, error) {
	if err := s.validate(h); err != nil {
		return nil, nil, err
	}
	b, err := s.backend.read(h.ID)
	if err != nil {
		return nil, nil, err
	}
	var r record
	defer func() { clear(r.Seed) }()
	if len(b) > 1024 {
		clear(b)
		return nil, nil, fmt.Errorf("oversized key record")
	}
	err = json.Unmarshal(b, &r)
	encoded, encodeErr := json.Marshal(r)
	defer clear(encoded)
	if err != nil || encodeErr != nil || !bytes.Equal(encoded, b) || r.Version != h.Version || r.Scope != h.Scope || r.ID != h.ID || len(r.Seed) != ed25519.SeedSize {
		clear(b)
		return nil, nil, fmt.Errorf("invalid private key record; explicit recovery required")
	}
	public, err := publicKey(r.Version, r.Seed)
	if err != nil || !bytes.Equal(public, h.PublicKey) {
		clear(b)
		return nil, nil, fmt.Errorf("private key/public handle mismatch")
	}
	return b, bytes.Clone(r.Seed), nil
}

// Signer never returns private bytes. Each signature reloads the protected
// record, so deletion or loss also invalidates previously opened handles.
func (s *Store) Signer(h Handle) (crypto.Signer, error) {
	b, k, err := s.load(h)
	if err != nil {
		return nil, err
	}
	clear(b)
	clear(k)
	h.PublicKey = bytes.Clone(h.PublicKey)
	return &signer{store: s, handle: h}, nil
}

// Delete removes only the exact validated handle. This is key destruction,
// not certificate revocation, and must be called by the owning lifecycle.
func (s *Store) Delete(h Handle) error {
	b, k, err := s.loadMaterial(h)
	if err != nil {
		return err
	}
	defer clear(b)
	clear(k)
	return s.backend.remove(h.ID, b)
}

type signer struct {
	store  *Store
	handle Handle
}

func (s *signer) Public() crypto.PublicKey { return ed25519.PublicKey(bytes.Clone(s.handle.PublicKey)) }
func (s *signer) Sign(_ io.Reader, message []byte, opts crypto.SignerOpts) ([]byte, error) {
	if opts == nil || opts.HashFunc() != crypto.Hash(0) || len(message) > 1<<20 {
		return nil, fmt.Errorf("only bounded pure Ed25519 signing is supported")
	}
	if o, ok := opts.(*ed25519.Options); ok && o.Context != "" {
		return nil, fmt.Errorf("Ed25519 context mode is unsupported")
	}
	b, k, err := s.store.load(s.handle)
	if err != nil {
		return nil, err
	}
	defer clear(b)
	defer clear(k)
	return ed25519.Sign(k, message), nil
}
