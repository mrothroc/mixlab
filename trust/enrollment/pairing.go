package enrollment

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const PairingVersion = "mixlab_enrollment_pairing_v1"

// PairingContext is the exact request-specific input to the TLS exporter.
// Network metadata and display names are not identity or approval evidence.
type PairingContext struct {
	Version     string     `json:"version"`
	Fingerprint string     `json:"fingerprint"`
	Audience    string     `json:"audience"`
	Purpose     Purpose    `json:"purpose"`
	Role        trust.Role `json:"role"`
	RequestID   string     `json:"request_id"`
	RequestHash string     `json:"request_hash"`
	ClientNonce []byte     `json:"client_nonce"`
	ServerNonce []byte     `json:"server_nonce"`
}

func (c PairingContext) Digest() ([32]byte, error) {
	if c.Version != PairingVersion || !digestOK(c.Fingerprint) || !validAudience(c.Audience) || c.Purpose.role() == "" || c.Role != c.Purpose.role() ||
		!certificates.ValidID(c.RequestID) || !digestOK(c.RequestHash) || len(c.ClientNonce) != 32 || len(c.ServerNonce) != 32 ||
		bytes.Equal(c.ClientNonce, make([]byte, 32)) || bytes.Equal(c.ServerNonce, make([]byte, 32)) {
		return [32]byte{}, fmt.Errorf("invalid enrollment pairing context")
	}
	b, err := json.Marshal(c)
	if err != nil {
		return [32]byte{}, err
	}
	return sha256.Sum256(b), nil
}

// PairingPresentation derives the public five-word SAS and a non-secret audit
// digest. Exporter bytes must come from the dedicated live transport port, never
// from a network request. Neither returned value is a confirmation or grant.
func PairingPresentation(c PairingContext, exporter []byte) (phrase, digest string, err error) {
	contextDigest, err := c.Digest()
	if err != nil {
		return "", "", err
	}
	if len(exporter) != 32 || bytes.Equal(exporter, make([]byte, 32)) {
		return "", "", fmt.Errorf("invalid enrollment exporter")
	}
	h := sha256.New()
	_, _ = h.Write([]byte("mixlab-enrollment-sas-v1"))
	_, _ = h.Write(exporter)
	_, _ = h.Write(contextDigest[:])
	var d [32]byte
	copy(d[:], h.Sum(nil))
	return identity.Confirmation(d), hex.EncodeToString(d[:]), nil
}
