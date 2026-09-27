package securekeys

import (
	"bytes"
	"errors"
	"testing"
)

func TestPendingKeys(t *testing.T) {
	s, _ := fileStore(t)
	pendingContract(t, s)
}

func pendingContract(t *testing.T, s *Store) {
	t.Helper()
	for _, envelope := range []bool{false, true} {
		p, err := s.PrepareSigning()
		if envelope {
			if p != nil {
				p.Close()
			}
			p, err = s.PrepareEnvelope()
		}
		if err != nil {
			t.Fatal(err)
		}
		h := p.Handle()
		if err := s.Inspect(h); !errors.Is(err, ErrMissing) {
			t.Fatal("prepared key already published", err)
		}
		changed := p.Handle()
		changed.PublicKey[0] ^= 1
		if bytes.Equal(p.Handle().PublicKey, changed.PublicKey) {
			t.Fatal("mutable candidate")
		}
		if err := p.Publish(); err != nil {
			t.Fatal(err)
		}
		if err := p.Publish(); err != nil {
			t.Fatal("same-key retry", err)
		}
		p.Close()
		if len(p.encoded) != 0 {
			t.Fatal("private material retained")
		}
		if err := p.Publish(); err == nil {
			t.Fatal("closed pending published")
		}
		if err := s.Inspect(h); err != nil {
			t.Fatal(err)
		}
		if err := s.Delete(h); err != nil {
			t.Fatal(err)
		}
	}
}

func TestPendingUncertainPublicationRetry(t *testing.T) {
	b := uncertainCreate{&fakeBackend{values: map[string][]byte{}}}
	s, err := newStore(b, "keychain", testScope)
	if err != nil {
		t.Fatal(err)
	}
	p, err := s.PrepareSigning()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if err := p.Publish(); err == nil {
		t.Fatal("expected uncertain write")
	}
	if err := p.Publish(); err != nil {
		t.Fatal("exact-key reconciliation", err)
	}
	if len(b.values) != 1 {
		t.Fatal("duplicate keys")
	}
}

func TestSelectedBackendRejectsUnknown(t *testing.T) {
	_, p := fileStore(t)
	if _, err := OpenSelected("unknown", p, testScope); err == nil {
		t.Fatal("unknown backend accepted")
	}
	s, err := OpenSelected("file", p, testScope)
	if err != nil || s.Backend() != "file" || s.Scope() != testScope {
		t.Fatal("explicit backend", err)
	}
}
