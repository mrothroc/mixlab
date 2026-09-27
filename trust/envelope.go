package trust

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/internal/credentialcrypto"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const EnvelopeVersion = "mixlab_credential_envelope_v1"
const EnvelopeLifetime = 15 * time.Minute

// EnvelopeBinding is supplied by the owning admitted-job context. A credential
// kind is opaque to trust; trust never parses the secret's authorization rules.
type EnvelopeBinding struct {
	Cluster        string `json:"cluster"`
	Node           string `json:"node"`
	Lease          string `json:"lease"`
	Job            string `json:"job"`
	Run            string `json:"run"`
	Worker         string `json:"worker"`
	Attempt        string `json:"attempt"`
	Audience       string `json:"audience"`
	CredentialKind string `json:"credential_kind"`
	IssuedAt       int64  `json:"issued_at"`
	ExpiresAt      int64  `json:"expires_at"`
}

type EnvelopePayload struct {
	Version       string          `json:"version"`
	Suite         string          `json:"suite"`
	Binding       EnvelopeBinding `json:"binding"`
	RecipientKey  []byte          `json:"recipient_key"`
	Nonce         string          `json:"nonce"`
	Encapsulation []byte          `json:"encapsulation"`
	Ciphertext    []byte          `json:"ciphertext"`
}

type CredentialEnvelope struct {
	Payload EnvelopePayload `json:"payload"`
	Proof   SignedProof     `json:"proof"`
}

func (b EnvelopeBinding) validate(a Anchor, now time.Time) error {
	for _, id := range []string{b.Cluster, b.Node, b.Lease, b.Job, b.Run, b.Worker, b.Attempt} {
		if !certificates.ValidID(id) {
			return fmt.Errorf("invalid credential binding identifier")
		}
	}
	if b.Cluster != a.Cluster() || !validText(b.Audience, 256) || !validText(b.CredentialKind, 128) {
		return fmt.Errorf("invalid credential envelope binding")
	}
	return timeWindow(b.IssuedAt, b.ExpiresAt, EnvelopeLifetime, now)
}

func envelopeAAD(p EnvelopePayload) ([]byte, error) {
	return canonical(struct {
		Version      string          `json:"version"`
		Suite        string          `json:"suite"`
		Binding      EnvelopeBinding `json:"binding"`
		RecipientKey []byte          `json:"recipient_key"`
		Nonce        string          `json:"nonce"`
	}{p.Version, p.Suite, p.Binding, p.RecipientKey, p.Nonce})
}

func envelopeRequest(p EnvelopePayload) (SignRequest, error) {
	digest, err := objectDigest(p)
	return SignRequest{Version: ProofVersion, Purpose: CredentialEnvelopePurpose, Digest: digest, Context: p.Binding.Job + "/" + p.Binding.Attempt, Audience: p.Binding.Audience}, err
}

// SealCredentialEnvelope authenticates the enrolled recipient and signs the
// encrypted transport value. The admitted-job owner sets expiry no later than
// its lease/job cleanup deadline. Only controller principals can seal in v1.
func SealCredentialEnvelope(a Anchor, v VerifiedSnapshot, controllerChain [][]byte, signer crypto.Signer, nodeChain [][]byte, binding EnvelopeBinding, secret []byte, now time.Time) (CredentialEnvelope, error) {
	var out CredentialEnvelope
	if err := binding.validate(a, now); err != nil {
		return out, err
	}
	if len(secret) == 0 || len(secret) > credentialcrypto.MaxPlaintext {
		return out, fmt.Errorf("credential secret size out of bounds")
	}
	controller, err := principal(a, controllerChain, v, now)
	if err != nil {
		return out, err
	}
	if controller.Role != Controller {
		return out, fmt.Errorf("only controllers seal v1 credential envelopes")
	}
	public, err := NodeEnvelopeKey(a, v, nodeChain, binding.Node, now)
	if err != nil {
		return out, err
	}
	for _, chain := range [][][]byte{controllerChain, nodeChain} {
		id, err := principal(a, chain, v, now)
		if err != nil {
			return out, err
		}
		if err := v.checkRevocation(id, 0); err != nil {
			return out, err
		}
		c, _, err := certificates.Parse(chain[0])
		if err != nil || binding.ExpiresAt > c.NotAfter.Unix() {
			return out, fmt.Errorf("envelope outlives principal")
		}
	}
	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		return out, err
	}
	p := EnvelopePayload{Version: EnvelopeVersion, Suite: credentialcrypto.SuiteID, Binding: binding, RecipientKey: public, Nonce: hex.EncodeToString(nonce)}
	aad, err := envelopeAAD(p)
	if err != nil {
		return out, err
	}
	p.Encapsulation, p.Ciphertext, err = credentialcrypto.Seal(public, []byte(EnvelopeVersion), aad, secret)
	if err != nil {
		return out, err
	}
	r, err := envelopeRequest(p)
	if err != nil {
		return out, err
	}
	proof, err := SignPrincipalProof(a, controllerChain, signer, v, r, now)
	if err != nil {
		return out, err
	}
	out = CredentialEnvelope{p, proof}
	if _, err := canonical(out); err != nil {
		return CredentialEnvelope{}, err
	}
	return out, nil
}

// EnvelopeDigest is the exact signed transport binding referenced by a NodeJob.
// Neither ciphertext nor plaintext belongs in an artifact or manifest.
func EnvelopeDigest(e CredentialEnvelope) (string, error) { return objectDigest(e) }

func DecodeEnvelope(raw []byte) (CredentialEnvelope, error) {
	var e CredentialEnvelope
	err := decodeCanonical(raw, &e)
	return e, err
}

// EnvelopeReplayGuard must atomically and durably reserve the digest in the
// owning job's journal before decryption. A failed/uncertain reservation rejects
// use; callers must not substitute a fresh in-memory guard after a restart.
type EnvelopeReplayGuard interface {
	ReserveEnvelope(binding EnvelopeBinding, digest string) error
}

// EnvelopeKeyOpener is implemented by the protected key adapter. The key is
// never sent to a worker. This port belongs to the node's credential-use broker.
type EnvelopeKeyOpener interface {
	PublicKey() []byte
	Open(info, aad, encapsulation, ciphertext []byte) ([]byte, error)
}

// OpenCredentialEnvelope requires the exact manifest digest, binding from
// admitted lease/job state and controller from the authenticated prepare call.
// Returned bytes stay inside the authorized credential-use broker: never child
// args/env/files, status, logs or crash reports. The broker owns secure cleanup.
func OpenCredentialEnvelope(a Anchor, v VerifiedSnapshot, e CredentialEnvelope, want EnvelopeBinding, wantDigest, authenticatedController string, nodeChain [][]byte, key EnvelopeKeyOpener, replay EnvelopeReplayGuard, now time.Time) ([]byte, error) {
	// Freeze mutable caller slices before checking and using the signed bytes.
	raw, err := canonical(e)
	if err != nil {
		return nil, err
	}
	e, err = DecodeEnvelope(raw)
	if err != nil {
		return nil, err
	}
	p := e.Payload
	if err := p.Binding.validate(a, now); err != nil {
		return nil, err
	}
	digest, err := EnvelopeDigest(e)
	if err != nil || digest != wantDigest || p.Binding != want || p.Version != EnvelopeVersion || p.Suite != credentialcrypto.SuiteID ||
		e.Proof.Evidence.Principal != authenticatedController || !certificates.ValidID(authenticatedController) || key == nil || replay == nil {
		return nil, fmt.Errorf("credential envelope does not match authenticated prepare and admitted job")
	}
	nonce, err := hex.DecodeString(p.Nonce)
	if err != nil || len(nonce) != 32 || hex.EncodeToString(nonce) != p.Nonce || len(p.Encapsulation) != 32 || len(p.Ciphertext) <= 16 || len(p.Ciphertext) > credentialcrypto.MaxPlaintext+16 {
		return nil, fmt.Errorf("invalid envelope encoding")
	}
	public, err := NodeEnvelopeKey(a, v, nodeChain, want.Node, now)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(public, p.RecipientKey) || !bytes.Equal(public, key.PublicKey()) {
		return nil, fmt.Errorf("envelope recipient key mismatch")
	}
	r, err := envelopeRequest(p)
	if err != nil {
		return nil, err
	}
	if _, err := VerifyProof(a, v, e.Proof, r, now); err != nil {
		return nil, err
	}
	aad, err := envelopeAAD(p)
	if err != nil {
		return nil, err
	}
	if err := replay.ReserveEnvelope(want, digest); err != nil {
		return nil, fmt.Errorf("envelope replay reservation failed: %w", err)
	}
	return key.Open([]byte(EnvelopeVersion), aad, p.Encapsulation, p.Ciphertext)
}
