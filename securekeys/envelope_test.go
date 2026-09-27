package securekeys

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/internal/credentialcrypto"
)

func envelopeContract(t *testing.T, s *Store) {
	t.Helper()
	h, err := s.GenerateEnvelope()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := s.Delete(h); err != nil {
			t.Error(err)
		}
	}()
	if h.Version != EnvelopeVersion {
		t.Fatal("wrong key profile")
	}
	if err := credentialcrypto.ValidatePublic(h.PublicKey); err != nil {
		t.Fatal(err)
	}
	opener, err := s.EnvelopeOpener(h)
	if err != nil {
		t.Fatal(err)
	}
	info, aad, pt := []byte("mixlab/test/v1"), []byte("node/job/attempt"), []byte("synthetic credential")
	enc, ct, err := credentialcrypto.Seal(h.PublicKey, info, aad, pt)
	if err != nil {
		t.Fatal(err)
	}
	got, err := opener.Open(info, aad, enc, ct)
	if err != nil || !bytes.Equal(got, pt) {
		t.Fatal("protected HPKE round trip", err)
	}
	pub := opener.PublicKey()
	pub[0] ^= 1
	if !bytes.Equal(opener.PublicKey(), h.PublicKey) {
		t.Fatal("public key aliases caller memory")
	}
	encoded, err := json.Marshal(h)
	if err != nil || bytes.Contains(encoded, []byte("seed")) || bytes.Contains(encoded, []byte("private")) {
		t.Fatal("secret in handle", err)
	}
	var restored Handle
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	if _, err := s.EnvelopeOpener(restored); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Handle){
		func(h *Handle) { h.Version = Version },
		func(h *Handle) { h.Scope = strings.Repeat("f", 64) },
		func(h *Handle) { h.ID = "../key" },
		func(h *Handle) { h.PublicKey = make([]byte, 32) },
		func(h *Handle) { h.Backend = "other" },
	} {
		bad := h
		mutate(&bad)
		if _, err := s.EnvelopeOpener(bad); err == nil {
			t.Fatal("bad handle accepted")
		}
		if err := s.Delete(bad); err == nil {
			t.Fatal("bad handle deleted key")
		}
	}
	if _, err := s.Signer(h); err == nil {
		t.Fatal("envelope key signs")
	}
	relabel := h
	relabel.Version = Version
	if _, err := s.Signer(relabel); err == nil {
		t.Fatal("envelope key relabeled as signing key")
	}
	signing, err := s.Generate()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := s.Delete(signing); err != nil {
			t.Error(err)
		}
	}()
	if signing.ID == h.ID || bytes.Equal(signing.PublicKey, h.PublicKey) {
		t.Fatal("key reuse")
	}
	if _, err := s.EnvelopeOpener(signing); err == nil {
		t.Fatal("signing key decrypts")
	}
	relabel = signing
	relabel.Version = EnvelopeVersion
	if _, err := s.EnvelopeOpener(relabel); err == nil {
		t.Fatal("signing key relabeled as envelope key")
	}
	h2, err := s.GenerateEnvelope()
	if err != nil {
		t.Fatal(err)
	}
	opener2, err := s.EnvelopeOpener(h2)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := opener2.Open(info, aad, enc, ct); err == nil {
		t.Fatal("cross-key ciphertext opened")
	}
	if err := s.Delete(h2); err != nil {
		t.Fatal(err)
	}
	if _, err := opener2.Open(info, aad, enc, ct); !errors.Is(err, ErrMissing) {
		t.Fatal("deleted key still usable", err)
	}
	if _, err := s.EnvelopeOpener(h2); !errors.Is(err, ErrMissing) {
		t.Fatal("deleted key recreated", err)
	}
	for _, changed := range []string{"info", "aad", "ciphertext"} {
		i, a, c := bytes.Clone(info), bytes.Clone(aad), bytes.Clone(ct)
		switch changed {
		case "info":
			i[0] ^= 1
		case "aad":
			a[0] ^= 1
		case "ciphertext":
			c[0] ^= 1
		}
		if out, err := opener.Open(i, a, enc, c); err == nil || len(out) != 0 {
			t.Fatal("tampered envelope opened", changed)
		}
	}
}

func TestFileEnvelopeContract(t *testing.T) { s, _ := fileStore(t); envelopeContract(t, s) }

func TestKeychainEnvelopeContractFake(t *testing.T) {
	b := &fakeBackend{values: map[string][]byte{}}
	s, err := newStore(b, "keychain", testScope)
	if err != nil {
		t.Fatal(err)
	}
	envelopeContract(t, s)
	h, err := s.GenerateEnvelope()
	if err != nil {
		t.Fatal(err)
	}
	o, err := s.EnvelopeOpener(h)
	if err != nil {
		t.Fatal(err)
	}
	b.failure = ErrUnavailable
	if _, err := o.Open(nil, nil, nil, nil); !errors.Is(err, ErrUnavailable) {
		t.Fatal("unavailable key not rechecked", err)
	}
	if _, err := s.GenerateEnvelope(); !errors.Is(err, ErrUnavailable) {
		t.Fatal("backend failure fallback", err)
	}
}

func TestEnvelopeReopenAndUncertainPublication(t *testing.T) {
	s, p := fileStore(t)
	h, err := s.GenerateEnvelope()
	if err != nil {
		t.Fatal(err)
	}
	reopened, err := OpenFile(p, testScope)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reopened.EnvelopeOpener(h); err != nil {
		t.Fatal(err)
	}
	if err := reopened.Delete(h); err != nil {
		t.Fatal(err)
	}
	b := uncertainCreate{&fakeBackend{values: map[string][]byte{}}}
	s, err = newStore(b, "keychain", testScope)
	if err != nil {
		t.Fatal(err)
	}
	h, err = s.GenerateEnvelope()
	if err == nil || h.ID == "" {
		t.Fatal("lost uncertain-publication handle")
	}
	if err := s.Delete(h); err != nil || len(b.values) != 0 {
		t.Fatal("orphan after reconciliation", err)
	}
}
