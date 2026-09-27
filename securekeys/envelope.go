package securekeys

import (
	"bytes"
	"fmt"

	"github.com/mrothroc/mixlab/internal/credentialcrypto"
)

// GenerateEnvelope creates an independent X25519 key, never a conversion of a
// TLS/signing key. Publication-error reconciliation is identical to Generate.
func (s *Store) GenerateEnvelope() (Handle, error) { return s.generate(EnvelopeVersion) }

type EnvelopeOpener struct {
	store  *Store
	handle Handle
}

func (s *Store) EnvelopeOpener(h Handle) (*EnvelopeOpener, error) {
	if h.Version != EnvelopeVersion {
		return nil, fmt.Errorf("not an envelope-key handle")
	}
	b, seed, err := s.loadMaterial(h)
	if err != nil {
		return nil, err
	}
	clear(b)
	clear(seed)
	h.PublicKey = bytes.Clone(h.PublicKey)
	return &EnvelopeOpener{store: s, handle: h}, nil
}

func (o *EnvelopeOpener) PublicKey() []byte { return bytes.Clone(o.handle.PublicKey) }

// Open is a crypto-adapter port, not an application credential-use API. The
// trust caller must verify envelope signature, binding and expiry first; the
// job-owning broker must authorize use and protect/delete the returned secret.
func (o *EnvelopeOpener) Open(info, aad, enc, ciphertext []byte) ([]byte, error) {
	b, seed, err := o.store.loadMaterial(o.handle)
	if err != nil {
		return nil, err
	}
	defer clear(b)
	defer clear(seed)
	return credentialcrypto.Open(seed, info, aad, enc, ciphertext)
}
