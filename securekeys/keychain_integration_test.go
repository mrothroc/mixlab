//go:build darwin && cgo && keychaintest

package securekeys

import (
	"crypto"
	"errors"
	"path/filepath"
	"testing"
)

func TestDisposableNativeKeychain(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	s, cleanup, err := disposableKeychain(filepath.Join(dir, "test.keychain"), testScope)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := cleanup(); err != nil {
			t.Error(err)
		}
	})
	storeContract(t, s)
	pendingContract(t, s)
	envelopeContract(t, s)
	envelopeHandle, err := s.GenerateEnvelope()
	if err != nil {
		t.Fatal(err)
	}
	opener, err := s.EnvelopeOpener(envelopeHandle)
	if err != nil {
		t.Fatal(err)
	}
	h, err := s.Generate()
	if err != nil {
		t.Fatal(err)
	}
	k, err := s.Signer(h)
	if err != nil {
		t.Fatal(err)
	}
	if err := lockTestKeychain(s); err != nil {
		t.Fatal(err)
	}
	if _, err := k.Sign(nil, []byte("locked"), crypto.Hash(0)); !errors.Is(err, ErrUnavailable) {
		t.Fatal("locked Keychain did not fail closed", err)
	}
	if _, err := opener.Open(nil, nil, nil, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatal("locked envelope Keychain did not fail closed", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Generate(); !errors.Is(err, ErrUnavailable) {
		t.Fatal("closed store recreated keys", err)
	}
}
