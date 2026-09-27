package certificates

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/internal/credentialcrypto"
)

const Node Role = "node"
const Worker Role = "worker"

// WorkloadLifetime is a ceiling, not a default: issuance is additionally bound
// to the admitted job's runtime and cleanup deadline.
const WorkloadLifetime = 7 * 24 * time.Hour
const NodeBindingVersion = "mixlab_node_binding_v1"
const WorkloadBindingVersion = "mixlab_ddp_workload_binding_v1"

const bindingURIPrefix = "urn:mixlab:binding:v1:"

// Bound roles use the standard critical SAN extension: identity URI first,
// versioned binding URI second. No private extension OID is required.
func subjectAltNameOID() asn1.ObjectIdentifier { return asn1.ObjectIdentifier{2, 5, 29, 17} }

func bindingURI(role Role, raw []byte) *url.URL {
	return &url.URL{Scheme: "urn", Opaque: "mixlab:binding:v1:" + string(role) + ":" + base64.RawURLEncoding.EncodeToString(raw)}
}

type NodeBinding struct {
	Version     string `json:"version"`
	EnvelopeKey []byte `json:"envelope_key"`
}

// WorkloadBinding contains only the R1.1 DDP identity contract. DiLoCo and
// recovery scopes require a separately versioned profile, not optional fields.
type WorkloadBinding struct {
	Version        string `json:"version"`
	Cluster        string `json:"cluster"`
	Role           Role   `json:"role"`
	Principal      string `json:"principal"`
	Participant    string `json:"participant"`
	Run            string `json:"run"`
	Job            string `json:"job"`
	Lease          string `json:"lease"`
	ManifestHash   string `json:"manifest_hash"`
	Attempt        string `json:"attempt"`
	Audience       string `json:"audience"`
	IssuedAt       int64  `json:"issued_at"`
	ExpiresAt      int64  `json:"expires_at"`
	Group          string `json:"group"`
	Generation     uint64 `json:"generation"`
	MembershipHash string `json:"membership_hash"`
	Member         string `json:"member"`
	Rank           int    `json:"rank"`
}

func (b WorkloadBinding) validate(id Identity, c *x509.Certificate) error {
	for _, value := range []string{b.Cluster, b.Principal, b.Participant, b.Run, b.Job, b.Lease, b.Attempt, b.Group, b.Member} {
		if !ValidID(value) {
			return fmt.Errorf("invalid workload binding identifier")
		}
	}
	for _, digest := range []string{b.MembershipHash, b.ManifestHash} {
		h, err := hex.DecodeString(digest)
		if err != nil || len(h) != 32 || hex.EncodeToString(h) != digest {
			return fmt.Errorf("invalid workload scope digest")
		}
	}
	if b.Generation == 0 || b.Rank < 0 || b.Rank > 1<<20 {
		return fmt.Errorf("invalid workload membership binding")
	}
	if len(b.Audience) == 0 || len(b.Audience) > 256 {
		return fmt.Errorf("invalid workload audience")
	}
	for _, r := range b.Audience {
		if r < 33 || r > 126 {
			return fmt.Errorf("invalid workload audience")
		}
	}
	if b.Version != WorkloadBindingVersion || b.Cluster != id.Cluster || b.Role != Worker || b.Role != id.Role || b.Principal != id.Principal ||
		b.IssuedAt <= 0 || b.ExpiresAt <= b.IssuedAt || b.ExpiresAt-b.IssuedAt > int64(WorkloadLifetime/time.Second) ||
		c.NotBefore.Unix() != b.IssuedAt || c.NotAfter.Unix() != b.ExpiresAt {
		return fmt.Errorf("workload binding/profile mismatch")
	}
	return nil
}

func bindingJSON(raw []byte, out any) error {
	if len(raw) > 4096 {
		return fmt.Errorf("oversized certificate binding")
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return err
	}
	b, err := json.Marshal(out)
	if err != nil || !bytes.Equal(b, raw) {
		return fmt.Errorf("noncanonical certificate binding")
	}
	return nil
}

func parseBindings(c *x509.Certificate, id *Identity) error {
	if len(c.UnhandledCriticalExtensions) != 0 {
		return fmt.Errorf("unhandled critical certificate extension")
	}
	if id.Role != Node && id.Role != Worker {
		if len(c.URIs) != 1 {
			return fmt.Errorf("binding on wrong certificate role")
		}
		return nil
	}
	if len(c.URIs) != 2 {
		return fmt.Errorf("bound role requires exactly one identity and one binding URI")
	}
	// Compare the raw SAN too: the runtime may ignore unsupported GeneralNames.
	var san *pkix.Extension
	for i := range c.Extensions {
		if c.Extensions[i].Id.Equal(subjectAltNameOID()) {
			if san != nil {
				return fmt.Errorf("duplicate subject alternative name")
			}
			san = &c.Extensions[i]
		}
	}
	expected, err := asn1.Marshal([]asn1.RawValue{
		{Class: 2, Tag: 6, Bytes: []byte(c.URIs[0].String())},
		{Class: 2, Tag: 6, Bytes: []byte(c.URIs[1].String())},
	})
	if err != nil || san == nil || !san.Critical || !bytes.Equal(san.Value, expected) {
		return fmt.Errorf("bound role requires canonical critical URI-only SAN")
	}
	u := c.URIs[1].String()
	prefix := bindingURIPrefix + string(id.Role) + ":"
	if !strings.HasPrefix(u, prefix) || len(u) > 6000 {
		return fmt.Errorf("unsupported binding URI role, version, or size")
	}
	raw, err := base64.RawURLEncoding.Strict().DecodeString(strings.TrimPrefix(u, prefix))
	if err != nil || bindingURI(id.Role, raw).String() != u {
		return fmt.Errorf("noncanonical binding URI")
	}
	switch id.Role {
	case Node:
		var b NodeBinding
		if err := bindingJSON(raw, &b); err != nil {
			return err
		}
		if b.Version != NodeBindingVersion {
			return fmt.Errorf("unsupported node binding version")
		}
		if err := credentialcrypto.ValidatePublic(b.EnvelopeKey); err != nil {
			return err
		}
		if bytes.Equal(c.PublicKey.(ed25519.PublicKey), b.EnvelopeKey) {
			return fmt.Errorf("node signing/envelope key reuse")
		}
		id.EnvelopeKey = bytes.Clone(b.EnvelopeKey)
	case Worker:
		var b WorkloadBinding
		if err := bindingJSON(raw, &b); err != nil {
			return err
		}
		if err := b.validate(*id, c); err != nil {
			return err
		}
		id.Workload = &b
	}
	return nil
}

func IssueNode(a Anchor, issuer []byte, signer crypto.Signer, principal string, public crypto.PublicKey, envelopeKey []byte, now time.Time) ([]byte, error) {
	if err := credentialcrypto.ValidatePublic(envelopeKey); err != nil {
		return nil, err
	}
	if p, ok := public.(ed25519.PublicKey); ok && bytes.Equal(p, envelopeKey) {
		return nil, fmt.Errorf("node signing and envelope keys must be separate")
	}
	return issueBound(a, issuer, signer, Node, principal, public, NodeBinding{NodeBindingVersion, bytes.Clone(envelopeKey)}, now)
}

// jobDeadline is supplied by the owning admitted-job workflow, including its
// cleanup allowance. Never derive it from an untrusted certificate request.
func IssueWorkload(a Anchor, issuer []byte, signer crypto.Signer, public crypto.PublicKey, binding WorkloadBinding, jobDeadline, now time.Time) ([]byte, error) {
	if binding.IssuedAt != now.UTC().Truncate(time.Second).Unix() || binding.ExpiresAt > jobDeadline.Unix() {
		return nil, fmt.Errorf("workload exceeds admitted job lifetime")
	}
	return issueBound(a, issuer, signer, Worker, binding.Principal, public, binding, now)
}

func issueBound(a Anchor, issuer []byte, signer crypto.Signer, role Role, principal string, public crypto.PublicKey, binding any, now time.Time) ([]byte, error) {
	parent, _, err := Verify(a, issuer, nil, Issuer, now)
	if err != nil {
		return nil, err
	}
	if err := MatchSigner(parent, signer); err != nil {
		return nil, err
	}
	c, err := template(Principal, a.cluster, role, principal, public, now)
	if err != nil {
		return nil, err
	}
	fp, _ := Fingerprint(public)
	parentFP, _ := Fingerprint(parent.PublicKey)
	if fp == a.fingerprint || fp == parentFP {
		return nil, fmt.Errorf("CA and subordinate keys must be distinct")
	}
	if b, ok := binding.(WorkloadBinding); ok {
		c.NotBefore, c.NotAfter = time.Unix(b.IssuedAt, 0), time.Unix(b.ExpiresAt, 0)
		if err := b.validate(Identity{Cluster: a.cluster, Role: role, Principal: principal}, c); err != nil {
			return nil, err
		}
	} else {
		if c.NotBefore.Before(parent.NotBefore) {
			c.NotBefore = parent.NotBefore
		}
		if c.NotAfter.After(parent.NotAfter) {
			c.NotAfter = parent.NotAfter
		}
	}
	if c.NotBefore.Before(parent.NotBefore) || c.NotAfter.After(parent.NotAfter) || !c.NotAfter.After(now) {
		return nil, fmt.Errorf("bound lifetime exceeds issuer")
	}
	b, err := json.Marshal(binding)
	if err != nil {
		return nil, err
	}
	c.URIs = append(c.URIs, bindingURI(role, b))
	return x509.CreateCertificate(rand.Reader, c, parent, public, signer)
}
