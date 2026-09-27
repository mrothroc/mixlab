package enrollment

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const interactiveVersion = "mixlab_interactive_enrollment_request_v1"

type InteractiveRequest struct {
	Version string `json:"version"`
	Window  string `json:"window"`
	Policy  Policy `json:"policy"`
	PrincipalRequest
}
type SignedInteractiveRequest struct {
	Request   InteractiveRequest `json:"request"`
	Signature []byte             `json:"signature"`
}

// NewInteractiveRequest requires the root already accepted under the explicit
// policy. Window metadata alone never establishes the pin or user confirmation.
func NewInteractiveRequest(a trust.Anchor, w Window, purpose Purpose, key crypto.Signer, envelope []byte, now time.Time) (SignedInteractiveRequest, error) {
	if err := w.validate(); err != nil {
		return SignedInteractiveRequest{}, err
	}
	if a.Cluster() != w.Cluster || a.Fingerprint() != w.Fingerprint || w.Closed || now.Unix() < w.Created || now.Unix() >= w.Expires || !w.allows(purpose) {
		return SignedInteractiveRequest{}, fmt.Errorf("window differs from accepted root/purpose or is closed")
	}
	if key == nil {
		return SignedInteractiveRequest{}, fmt.Errorf("local principal signer required")
	}
	pub, ok := key.Public().(ed25519.PublicKey)
	if !ok {
		return SignedInteractiveRequest{}, fmt.Errorf("Ed25519 principal key required")
	}
	r := InteractiveRequest{interactiveVersion, w.ID, w.Policy, PrincipalRequest{Cluster: w.Cluster, Fingerprint: w.Fingerprint, Endpoint: w.Endpoint, Audience: w.Audience, Purpose: purpose, Role: purpose.role(), PublicKey: bytes.Clone(pub), EnvelopeKey: bytes.Clone(envelope), Nonce: make([]byte, 32)}}
	if _, err := rand.Read(r.Nonce); err != nil {
		return SignedInteractiveRequest{}, err
	}
	if err := r.validate(); err != nil {
		return SignedInteractiveRequest{}, err
	}
	d := hash(interactiveVersion, r)
	signature, err := key.Sign(rand.Reader, d[:], crypto.Hash(0))
	if err != nil {
		return SignedInteractiveRequest{}, err
	}
	result := SignedInteractiveRequest{r, signature}
	return result, result.verify()
}
func (w Window) allows(p Purpose) bool {
	for _, v := range w.Purposes {
		if v == p {
			return true
		}
	}
	return false
}
func (r InteractiveRequest) validate() error {
	if r.Version != interactiveVersion || !certificates.ValidID(r.Window) || (r.Policy != TrustedLAN && r.Policy != Verified) {
		return fmt.Errorf("invalid interactive enrollment request")
	}
	return r.PrincipalRequest.validate()
}
func (r SignedInteractiveRequest) verify() error {
	if err := r.Request.validate(); err != nil {
		return err
	}
	d := hash(interactiveVersion, r.Request)
	if !ed25519.Verify(ed25519.PublicKey(r.Request.PublicKey), d[:], r.Signature) {
		return fmt.Errorf("invalid interactive enrollment proof of possession")
	}
	return nil
}
