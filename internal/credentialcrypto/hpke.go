// Package credentialcrypto fixes the R1.1 HPKE suite. It owns no identities,
// authorization, replay journal, or storage. CIRCL implements RFC 9180.
package credentialcrypto

import (
	"crypto/ecdh"
	"crypto/rand"
	"fmt"
	"io"

	"github.com/cloudflare/circl/hpke"
)

const MaxPlaintext = 64 << 10
const MaxBinding = 4096
const SuiteID = "DHKEM_X25519_HKDF_SHA256/HKDF_SHA256/ChaCha20Poly1305"

var suite = hpke.NewSuite(hpke.KEM_X25519_HKDF_SHA256, hpke.KDF_HKDF_SHA256, hpke.AEAD_ChaCha20Poly1305)

// ValidatePublic rejects noncanonical or low-order X25519 public keys before
// binding one into an identity. The fixed private input is public test material,
// used only for the runtime's low-order-point check, never to encrypt a secret.
func ValidatePublic(public []byte) error {
	if len(public) != 32 || public[31]&0x80 != 0 {
		return fmt.Errorf("invalid X25519 public key")
	}
	// RFC 7748 accepts noncanonical field elements; identity bindings do not.
	prime := [32]byte{0xed}
	for i := 1; i < 31; i++ {
		prime[i] = 0xff
	}
	prime[31] = 0x7f
	less := false
	for i := 31; i >= 0; i-- {
		if public[i] < prime[i] {
			less = true
			break
		}
		if public[i] > prime[i] {
			break
		}
	}
	if !less {
		return fmt.Errorf("noncanonical X25519 public key")
	}
	pk, err := ecdh.X25519().NewPublicKey(public)
	if err != nil {
		return err
	}
	probe, err := ecdh.X25519().NewPrivateKey(make([]byte, 32))
	if err != nil {
		return err
	}
	if _, err := probe.ECDH(pk); err != nil {
		return fmt.Errorf("invalid X25519 public key")
	}
	return nil
}

func Seal(public, info, aad, plaintext []byte) ([]byte, []byte, error) {
	return seal(public, info, aad, plaintext, rand.Reader)
}

func seal(public, info, aad, plaintext []byte, random io.Reader) ([]byte, []byte, error) {
	if len(plaintext) == 0 || len(plaintext) > MaxPlaintext || len(info) > MaxBinding || len(aad) > MaxBinding {
		return nil, nil, fmt.Errorf("HPKE input bounds exceeded")
	}
	if err := ValidatePublic(public); err != nil {
		return nil, nil, err
	}
	pk, err := hpke.KEM_X25519_HKDF_SHA256.Scheme().UnmarshalBinaryPublicKey(public)
	if err != nil {
		return nil, nil, err
	}
	sender, err := suite.NewSender(pk, info)
	if err != nil {
		return nil, nil, err
	}
	enc, sealer, err := sender.Setup(random)
	if err != nil {
		return nil, nil, err
	}
	ciphertext, err := sealer.Seal(plaintext, aad)
	return enc, ciphertext, err
}

// Open is used only inside the protected key adapter. No HPKE context or
// private-key object is returned to callers.
func Open(private, info, aad, enc, ciphertext []byte) ([]byte, error) {
	if len(private) != 32 || len(enc) != 32 || len(ciphertext) <= 16 || len(ciphertext) > MaxPlaintext+16 || len(info) > MaxBinding || len(aad) > MaxBinding {
		return nil, fmt.Errorf("HPKE input bounds exceeded")
	}
	if err := ValidatePublic(enc); err != nil {
		return nil, err
	}
	sk, err := hpke.KEM_X25519_HKDF_SHA256.Scheme().UnmarshalBinaryPrivateKey(private)
	if err != nil {
		return nil, err
	}
	receiver, err := suite.NewReceiver(sk, info)
	if err != nil {
		return nil, err
	}
	opener, err := receiver.Setup(enc)
	if err != nil {
		return nil, err
	}
	return opener.Open(ciphertext, aad)
}
