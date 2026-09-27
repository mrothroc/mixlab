// Package enrollment owns enrollment approval and issuance, not HTTP/TLS,
// discovery, node leases, or worker launch. No public listener is exposed here.
package enrollment

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/internal/credentialcrypto"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const invitationVersion = "mixlab_enrollment_invitation_v1"
const requestVersion = "mixlab_enrollment_request_v1"
const maxInvitationTTL = time.Hour

type Purpose string

const (
	NodeEnrollment        Purpose = "node-enrollment"
	ControllerEnrollment  Purpose = "controller-enrollment"
	CoordinatorEnrollment Purpose = "coordinator-enrollment"
)

func (p Purpose) role() trust.Role {
	switch p {
	case NodeEnrollment:
		return trust.Node
	case ControllerEnrollment:
		return trust.Controller
	case CoordinatorEnrollment:
		return trust.Coordinator
	}
	return ""
}

// Invitation is a protected, one-use provisioning-file payload. Only the
// verifier is retained by the authority. Do not put these bytes in argv/env.
type Invitation struct {
	Version     string     `json:"version"`
	ID          string     `json:"id"`
	Cluster     string     `json:"cluster"`
	Root        []byte     `json:"root"`
	Fingerprint string     `json:"fingerprint"`
	Wordlist    string     `json:"wordlist_version"`
	Endpoint    string     `json:"endpoint"`
	Audience    string     `json:"audience"`
	Purpose     Purpose    `json:"purpose"`
	Role        trust.Role `json:"role"`
	Secret      []byte     `json:"secret"`
	IssuedAt    int64      `json:"issued_at"`
	ExpiresAt   int64      `json:"expires_at"`
	UseLimit    int        `json:"use_limit"`
}

func (Invitation) String() string   { return "[protected enrollment invitation]" }
func (Invitation) GoString() string { return "[protected enrollment invitation]" }

// Clear erases the caller-owned bearer secret after protected publication or
// consumption. Public identity metadata need not be secret.
func (i *Invitation) Clear() { clear(i.Secret); i.Secret = nil }

// ValidateTarget must run before any provisioning secret is sent. The transport
// must separately authenticate the authority under this pinned root and the
// root-signed endpoint set. Discovery data is not a source of pins.
func (i Invitation) ValidateTarget(endpoint, audience string, now time.Time) (trust.Anchor, error) {
	var zero trust.Anchor
	if i.Version != invitationVersion || !certificates.ValidID(i.ID) || i.Purpose.role() == "" || i.Role != i.Purpose.role() ||
		i.UseLimit != 1 || i.Wordlist != identity.WordlistVersion || len(i.Secret) != 32 || bytes.Equal(i.Secret, make([]byte, 32)) ||
		i.IssuedAt <= 0 || i.IssuedAt > now.Unix() || i.ExpiresAt <= now.Unix() || i.ExpiresAt-i.IssuedAt > int64(maxInvitationTTL/time.Second) ||
		i.Endpoint != endpoint || i.Audience != audience || !validEndpoint(endpoint) || !validAudience(audience) {
		return zero, fmt.Errorf("invalid, expired, or mismatched provisioning file")
	}
	a, err := trust.PinRoot(i.Root, i.Fingerprint, now)
	if err != nil || a.Cluster() != i.Cluster {
		return zero, fmt.Errorf("provisioning root/cluster mismatch")
	}
	return a, nil
}

type Request struct {
	Version    string `json:"version"`
	Invitation string `json:"invitation"`
	PrincipalRequest
}

// PrincipalRequest is the shared, locally generated key request. Approval
// references and signature domains remain distinct for files and live windows.
type PrincipalRequest struct {
	Cluster     string     `json:"cluster"`
	Fingerprint string     `json:"fingerprint"`
	Endpoint    string     `json:"endpoint"`
	Audience    string     `json:"audience"`
	Purpose     Purpose    `json:"purpose"`
	Role        trust.Role `json:"role"`
	PublicKey   []byte     `json:"public_key"`
	EnvelopeKey []byte     `json:"envelope_key"`
	Nonce       []byte     `json:"nonce"`
}

type SignedRequest struct {
	Request   Request `json:"request"`
	Signature []byte  `json:"signature"`
}

// NewRequest uses the local protected signer. The envelope public key must
// come from an independently generated local X25519 handle for a node.
func NewRequest(i Invitation, key crypto.Signer, envelope []byte, now time.Time) (SignedRequest, error) {
	if _, err := i.ValidateTarget(i.Endpoint, i.Audience, now); err != nil {
		return SignedRequest{}, err
	}
	if key == nil {
		return SignedRequest{}, fmt.Errorf("missing principal key")
	}
	pub, ok := key.Public().(ed25519.PublicKey)
	if !ok {
		return SignedRequest{}, fmt.Errorf("principal key must be Ed25519")
	}
	r := Request{requestVersion, i.ID, PrincipalRequest{i.Cluster, i.Fingerprint, i.Endpoint, i.Audience, i.Purpose, i.Role,
		bytes.Clone(pub), bytes.Clone(envelope), make([]byte, 32)}}
	if _, err := rand.Read(r.Nonce); err != nil {
		return SignedRequest{}, err
	}
	if err := r.validate(); err != nil {
		return SignedRequest{}, err
	}
	digest := hash(requestVersion, r)
	sig, err := key.Sign(rand.Reader, digest[:], crypto.Hash(0))
	if err != nil {
		return SignedRequest{}, err
	}
	s := SignedRequest{r, sig}
	return s, s.verify()
}

func (r Request) validate() error {
	if r.Version != requestVersion || !certificates.ValidID(r.Invitation) {
		return fmt.Errorf("invalid provisioning request")
	}
	return r.PrincipalRequest.validate()
}

func (r PrincipalRequest) validate() error {
	if !certificates.ValidID(r.Cluster) || !digestOK(r.Fingerprint) ||
		r.Role == "" || r.Role != r.Purpose.role() || !validEndpoint(r.Endpoint) || !validAudience(r.Audience) || len(r.PublicKey) != 32 || len(r.Nonce) != 32 {
		return fmt.Errorf("invalid enrollment request")
	}
	if r.Role == trust.Node {
		if err := credentialcrypto.ValidatePublic(r.EnvelopeKey); err != nil {
			return err
		}
		if bytes.Equal(r.PublicKey, r.EnvelopeKey) {
			return fmt.Errorf("node key purposes must be separate")
		}
	} else if len(r.EnvelopeKey) != 0 {
		return fmt.Errorf("envelope key on non-node request")
	}
	return nil
}

func (s SignedRequest) verify() error {
	if err := s.Request.validate(); err != nil {
		return err
	}
	d := hash(requestVersion, s.Request)
	if !ed25519.Verify(ed25519.PublicKey(s.Request.PublicKey), d[:], s.Signature) {
		return fmt.Errorf("invalid enrollment proof of possession")
	}
	return nil
}

func hash(domain string, v any) [32]byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic("enrollment canonical struct is not serializable")
	}
	return sha256.Sum256(append([]byte(domain+"\x00"), b...))
}

func digestText(domain string, v any) string { d := hash(domain, v); return hex.EncodeToString(d[:]) }
func digestOK(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && hex.EncodeToString(b) == s
}
