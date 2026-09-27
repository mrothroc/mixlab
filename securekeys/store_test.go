package securekeys

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
)

const testScope = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func fileStore(t *testing.T) (*Store, statehome.Path) {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	s, err := OpenFile(p, testScope)
	if err != nil {
		t.Fatal(err)
	}
	return s, p
}

func storeContract(t *testing.T, s *Store) {
	t.Helper()
	h, err := s.Generate()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := s.backend.read(h.ID)
	if err != nil {
		t.Fatal(err)
	}
	defer clear(raw)
	if err := s.backend.create(h.ID, raw); !errors.Is(err, ErrExists) {
		t.Fatal("existing key overwritten", err)
	}
	k, err := s.Signer(h)
	if err != nil {
		t.Fatal(err)
	}
	message := []byte("test domain-bound digest")
	sig, err := k.Sign(nil, message, crypto.Hash(0))
	if err != nil || !ed25519.Verify(ed25519.PublicKey(h.PublicKey), message, sig) {
		t.Fatalf("sign: %v", err)
	}
	encoded, err := json.Marshal(h)
	if err != nil || bytes.Contains(encoded, []byte("seed")) || bytes.Contains(encoded, []byte("private")) {
		t.Fatal("handle exposes secret")
	}
	var restored Handle
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Signer(restored); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*Handle){
		"scope":        func(h *Handle) { h.Scope = strings.Repeat("f", 64) },
		"id traversal": func(h *Handle) { h.ID = "../other" },
		"version":      func(h *Handle) { h.Version = "v0" },
		"backend":      func(h *Handle) { h.Backend = "other" },
		"public key":   func(h *Handle) { h.PublicKey = make([]byte, 32) },
	} {
		t.Run(name, func(t *testing.T) {
			bad := h
			mutate(&bad)
			if _, err := s.Signer(bad); err == nil {
				t.Fatal("accepted wrong handle")
			}
			if err := s.Delete(bad); err == nil {
				t.Fatal("deleted wrong handle")
			}
		})
	}
	if _, err := k.Sign(nil, message, crypto.SHA256); err == nil {
		t.Fatal("accepted hash variant")
	}
	if _, err := k.Sign(nil, message, &ed25519.Options{Context: "other"}); err == nil {
		t.Fatal("accepted context variant")
	}
	if _, err := k.Sign(nil, message, nil); err == nil {
		t.Fatal("accepted missing options")
	}
	// Signer/public-return values must not alias mutable caller handles.
	pub := k.Public().(ed25519.PublicKey)
	pub[0] ^= 1
	if !ed25519.Verify(k.Public().(ed25519.PublicKey), message, sig) {
		t.Fatal("mutable signer public key")
	}
	h2, err := s.Generate()
	if err != nil || h.ID == h2.ID || bytes.Equal(h.PublicKey, h2.PublicKey) {
		t.Fatal("keys not independent", err)
	}
	if err := s.Delete(h); err != nil {
		t.Fatal(err)
	}
	if _, err := k.Sign(nil, message, crypto.Hash(0)); !errors.Is(err, ErrMissing) {
		t.Fatal("deleted handle still signs", err)
	}
	if _, err := s.Signer(h); !errors.Is(err, ErrMissing) {
		t.Fatal("missing key recreated", err)
	}
	if _, err := s.Signer(h2); err != nil {
		t.Fatal("deleted unrelated key", err)
	}
	if err := s.Delete(h2); err != nil {
		t.Fatal(err)
	}
}

func TestFileStoreContract(t *testing.T) { s, _ := fileStore(t); storeContract(t, s) }

type fakeBackend struct {
	mu      sync.Mutex
	values  map[string][]byte
	failure error
}

func (b *fakeBackend) create(id string, v []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failure != nil {
		return b.failure
	}
	if _, ok := b.values[id]; ok {
		return ErrExists
	}
	b.values[id] = bytes.Clone(v)
	return nil
}
func (b *fakeBackend) read(id string) ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failure != nil {
		return nil, b.failure
	}
	v, ok := b.values[id]
	if !ok {
		return nil, ErrMissing
	}
	return bytes.Clone(v), nil
}
func (b *fakeBackend) remove(id string, expected []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.failure != nil {
		return b.failure
	}
	if !bytes.Equal(b.values[id], expected) {
		return errors.New("changed")
	}
	clear(b.values[id])
	delete(b.values, id)
	return nil
}

func TestKeychainBackendContractFake(t *testing.T) {
	b := &fakeBackend{values: map[string][]byte{}}
	s, err := newStore(b, "keychain", testScope)
	if err != nil {
		t.Fatal(err)
	}
	storeContract(t, s)
	b.failure = ErrUnavailable
	if _, err := s.Generate(); !errors.Is(err, ErrUnavailable) {
		t.Fatal("fallback on backend failure", err)
	}
}

func TestFileStoreCorruptionAndPermissionRejection(t *testing.T) {
	for _, kind := range []string{"truncated", "oversized", "mode", "symlink", "hardlink", "seed-mismatch", "scope"} {
		t.Run(kind, func(t *testing.T) {
			s, p := fileStore(t)
			h, err := s.Generate()
			if err != nil {
				t.Fatal(err)
			}
			name := filepath.Join(p.Dir(), keyName(h.ID))
			b, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "truncated":
				err = os.WriteFile(name, []byte("{"), 0600)
			case "oversized":
				err = os.WriteFile(name, make([]byte, 2048), 0600)
			case "mode":
				err = os.Chmod(name, 0644)
			case "symlink":
				err = os.Rename(name, name+"-real")
				if err == nil {
					err = os.Symlink(name+"-real", name)
				}
			case "hardlink":
				err = os.Link(name, name+"-link")
			case "seed-mismatch", "scope":
				var r record
				err = json.Unmarshal(b, &r)
				if kind == "scope" {
					r.Scope = strings.Repeat("f", 64)
				} else {
					r.Seed[0] ^= 1
				}
				if err == nil {
					b, err = json.Marshal(r)
				}
				if err == nil {
					err = os.WriteFile(name, b, 0600)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.Signer(h); err == nil {
				t.Fatal("unsafe/corrupt key accepted")
			}
			if err := s.Delete(h); err == nil {
				t.Fatal("unsafe/corrupt key silently removed")
			}
		})
	}
}

func TestFileStoreReopenAndCrossScope(t *testing.T) {
	s, p := fileStore(t)
	h, err := s.Generate()
	if err != nil {
		t.Fatal(err)
	}
	other, err := OpenFile(p, strings.Repeat("f", 64))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := other.Signer(h); err == nil {
		t.Fatal("cross-context handle accepted")
	}
	h.Scope = other.scope
	if _, err := other.Signer(h); err == nil {
		t.Fatal("relabeling handle moved key ownership")
	}
	h.Scope = testScope
	reopened, err := OpenFile(p, testScope)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reopened.Signer(h); err != nil {
		t.Fatal(err)
	}
	if err := reopened.Delete(h); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Signer(h); !errors.Is(err, ErrMissing) {
		t.Fatal(err)
	}
}

type uncertainCreate struct{ *fakeBackend }

func (b uncertainCreate) create(id string, value []byte) error {
	if err := b.fakeBackend.create(id, value); err != nil {
		return err
	}
	return errors.New("publication durability unknown")
}

func TestUncertainPublicationRetainsCleanupHandle(t *testing.T) {
	b := uncertainCreate{&fakeBackend{values: map[string][]byte{}}}
	s, err := newStore(b, "keychain", testScope)
	if err != nil {
		t.Fatal(err)
	}
	h, err := s.Generate()
	if err == nil || h.ID == "" {
		t.Fatal("uncertain publication lost its reconciliation handle")
	}
	if err := s.Delete(h); err != nil {
		t.Fatal(err)
	}
	if len(b.values) != 0 {
		t.Fatal("orphan key remained")
	}
}
