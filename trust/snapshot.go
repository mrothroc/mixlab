// Package trust owns cluster identity and proof validation. It grants no node,
// lease, job, run, or artifact operations and has no network or GPU dependency.
package trust

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type Anchor = certificates.Anchor
type Role = certificates.Role
type WorkloadBinding = certificates.WorkloadBinding

const (
	Authority              = certificates.Authority
	Controller             = certificates.Controller
	Coordinator            = certificates.Coordinator
	Node                   = certificates.Node
	Worker                 = certificates.Worker
	WorkloadBindingVersion = certificates.WorkloadBindingVersion
	SnapshotVersion        = "mixlab_trust_snapshot_v1"
	EndpointVersion        = "mixlab_authority_endpoints_v1"
	SnapshotLifetime       = 15 * time.Minute
	MaxTrustBytes          = 256 << 10
)

func PinRoot(der []byte, fingerprint string, now time.Time) (Anchor, error) {
	return certificates.Pin(der, fingerprint, now)
}
func RootFingerprint(public crypto.PublicKey) (string, error) {
	return certificates.Fingerprint(public)
}

type AuthorityEndpoints struct {
	Version   string   `json:"version"`
	Cluster   string   `json:"cluster"`
	Audience  string   `json:"audience"`
	URLs      []string `json:"urls"`
	IssuedAt  int64    `json:"issued_at"`
	ExpiresAt int64    `json:"expires_at"`
}

type SignedEndpoints struct {
	Payload   AuthorityEndpoints `json:"payload"`
	Signature []byte             `json:"signature"`
}

type Revocation struct {
	Kind            string `json:"kind"` // principal or certificate
	ID              string `json:"id"`
	Mode            string `json:"mode"` // prospective or compromise
	Reason          string `json:"reason"`
	FirstGeneration uint64 `json:"first_generation"`
}

type Snapshot struct {
	Version       string          `json:"version"`
	Cluster       string          `json:"cluster"`
	Generation    uint64          `json:"generation"`
	IssuedAt      int64           `json:"issued_at"`
	ExpiresAt     int64           `json:"expires_at"`
	Issuers       [][]byte        `json:"issuers"`
	EligibleRoles []Role          `json:"eligible_roles"`
	Endpoints     SignedEndpoints `json:"endpoints"`
	Revocations   []Revocation    `json:"revocations"`
}

type SignedSnapshot struct {
	Payload           Snapshot `json:"payload"`
	SignerCertificate []byte   `json:"signer_certificate"`
	Signature         []byte   `json:"signature"`
}

// VerifiedSnapshot cannot be fabricated or mutated through exported fields.
// A repository must persist canonical signed bytes plus its monotonic generation
// atomically before exposing an advanced view; this type does not own storage.
type VerifiedSnapshot struct {
	anchor Anchor
	signed SignedSnapshot
}

func (v VerifiedSnapshot) Generation() uint64     { return v.signed.Payload.Generation }
func (v VerifiedSnapshot) Bytes() ([]byte, error) { return canonical(v.signed) }

func canonical(value any) ([]byte, error) {
	b, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if len(b) > MaxTrustBytes {
		return nil, fmt.Errorf("trust object too large")
	}
	return b, nil
}

func decodeCanonical(body []byte, dst any) error {
	if len(body) > MaxTrustBytes {
		return fmt.Errorf("trust object too large")
	}
	if err := json.Unmarshal(body, dst); err != nil {
		return err
	}
	again, err := canonical(dst)
	if err != nil || !bytes.Equal(body, again) {
		return fmt.Errorf("noncanonical trust object")
	}
	return nil
}

func signedDigest(domain string, value any) ([]byte, error) {
	b, err := canonical(value)
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	_, _ = h.Write([]byte(domain + "\x00"))
	_, _ = h.Write(b)
	return h.Sum(nil), nil
}

func sign(domain string, value any, key crypto.Signer) ([]byte, error) {
	digest, err := signedDigest(domain, value)
	if err != nil {
		return nil, err
	}
	return key.Sign(rand.Reader, digest, crypto.Hash(0))
}

func verify(domain string, value any, key crypto.PublicKey, signature []byte) error {
	digest, err := signedDigest(domain, value)
	if err != nil {
		return err
	}
	pub, ok := key.(ed25519.PublicKey)
	if !ok || len(signature) != ed25519.SignatureSize || !ed25519.Verify(pub, digest, signature) {
		return fmt.Errorf("invalid trust signature")
	}
	return nil
}

func validText(s string, max int) bool {
	if len(s) == 0 || len(s) > max {
		return false
	}
	for _, c := range s {
		if c < 0x21 || c > 0x7e {
			return false
		}
	}
	return true
}

func timeWindow(issued, expires int64, lifetime time.Duration, now time.Time) error {
	if issued <= 0 || expires <= issued || expires-issued > int64(lifetime/time.Second) ||
		issued > now.Add(certificates.ClockSkew).Unix() || expires <= now.Unix() {
		return fmt.Errorf("invalid or stale trust time window")
	}
	return nil
}

func validateEndpoints(a Anchor, e SignedEndpoints, now time.Time) error {
	if err := checkEndpointBody(a, e.Payload, now); err != nil {
		return err
	}
	root, _, _ := certificates.Parse(a.DER())
	return verify(EndpointVersion, e.Payload, root.PublicKey, e.Signature)
}

func checkEndpointBody(a Anchor, p AuthorityEndpoints, now time.Time) error {
	if p.Version != EndpointVersion || p.Cluster != a.Cluster() || !validText(p.Audience, 256) || len(p.URLs) < 1 || len(p.URLs) > 8 {
		return fmt.Errorf("invalid authority endpoint set")
	}
	if err := timeWindow(p.IssuedAt, p.ExpiresAt, certificates.IssuerLifetime, now); err != nil {
		return err
	}
	for i, s := range p.URLs {
		u, err := url.Parse(s)
		if err != nil || len(s) > 2048 || u.Scheme != "https" || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" ||
			(u.Path != "" && u.Path != "/") || (i > 0 && p.URLs[i-1] >= s) {
			return fmt.Errorf("invalid or unsorted HTTPS authority endpoint")
		}
	}
	root, _, err := certificates.Parse(a.DER())
	if err != nil {
		return err
	}
	if _, err = certificates.Pin(a.DER(), a.Fingerprint(), now); err != nil {
		return err
	}
	if p.ExpiresAt > root.NotAfter.Unix() {
		return fmt.Errorf("endpoint set outlives root")
	}
	return nil
}

// SignAuthorityEndpoints is a purpose-specific root operation, never available
// through a TLS principal or general PrincipalSigner.
func SignAuthorityEndpoints(a Anchor, p AuthorityEndpoints, key crypto.Signer, now time.Time) (SignedEndpoints, error) {
	if err := checkEndpointBody(a, p, now); err != nil {
		return SignedEndpoints{}, err
	}
	p.URLs = append([]string(nil), p.URLs...)
	root, _, err := certificates.Parse(a.DER())
	if err != nil {
		return SignedEndpoints{}, err
	}
	if err = certificates.MatchSigner(root, key); err != nil {
		return SignedEndpoints{}, err
	}
	sig, err := sign(EndpointVersion, p, key)
	if err != nil {
		return SignedEndpoints{}, err
	}
	return SignedEndpoints{p, sig}, nil
}

func SignSnapshot(a Anchor, p Snapshot, signerDER []byte, key crypto.Signer, now time.Time) (SignedSnapshot, error) {
	c, _, err := certificates.Verify(a, signerDER, nil, certificates.SnapshotSigner, now)
	if err != nil {
		return SignedSnapshot{}, err
	}
	if err = certificates.MatchSigner(c, key); err != nil {
		return SignedSnapshot{}, err
	}
	// Include the exact signing certificate in the signed payload, not just an
	// unsigned hint attached after signing. The signature itself is excluded.
	// Reserve the fixed-size signature before checking the wire-size budget.
	s := SignedSnapshot{Payload: p, SignerCertificate: bytes.Clone(signerDER), Signature: make([]byte, ed25519.SignatureSize)}
	v, err := checkSnapshotBody(a, s, now)
	if err != nil {
		return SignedSnapshot{}, err
	}
	s = v.signed
	s.Signature, err = sign(SnapshotVersion, snapshotBody(s), key)
	return s, err
}

func snapshotBody(s SignedSnapshot) any {
	return struct {
		Payload           Snapshot `json:"payload"`
		SignerCertificate []byte   `json:"signer_certificate"`
	}{s.Payload, s.SignerCertificate}
}

func VerifySnapshot(a Anchor, s SignedSnapshot, now time.Time) (VerifiedSnapshot, error) {
	v, err := checkSnapshotBody(a, s, now)
	if err != nil {
		return VerifiedSnapshot{}, err
	}
	c, _, _ := certificates.Parse(v.signed.SignerCertificate)
	if err = verify(SnapshotVersion, snapshotBody(v.signed), c.PublicKey, v.signed.Signature); err != nil {
		return VerifiedSnapshot{}, err
	}
	return v, nil
}

func checkSnapshotBody(a Anchor, s SignedSnapshot, now time.Time) (VerifiedSnapshot, error) {
	// Clone all caller-owned slices to prevent post-validation mutation.
	body, err := canonical(s)
	if err != nil {
		return VerifiedSnapshot{}, err
	}
	var frozen SignedSnapshot
	if err = decodeCanonical(body, &frozen); err != nil {
		return VerifiedSnapshot{}, err
	}
	s = frozen
	p := s.Payload
	if p.Version != SnapshotVersion || p.Cluster != a.Cluster() || p.Generation == 0 || len(p.Issuers) < 1 || len(p.Issuers) > 8 || len(p.Revocations) > 4096 || len(p.EligibleRoles) < 1 || len(p.EligibleRoles) > 5 {
		return VerifiedSnapshot{}, fmt.Errorf("invalid trust snapshot shape")
	}
	if err = timeWindow(p.IssuedAt, p.ExpiresAt, SnapshotLifetime, now); err != nil {
		return VerifiedSnapshot{}, err
	}
	c, sid, err := certificates.Verify(a, s.SignerCertificate, nil, certificates.SnapshotSigner, now)
	if err != nil {
		return VerifiedSnapshot{}, err
	}
	if p.IssuedAt < c.NotBefore.Unix() || p.ExpiresAt > c.NotAfter.Unix() {
		return VerifiedSnapshot{}, fmt.Errorf("snapshot outlives signer")
	}
	if err = validateEndpoints(a, p.Endpoints, now); err != nil {
		return VerifiedSnapshot{}, err
	}
	if p.ExpiresAt > p.Endpoints.Payload.ExpiresAt {
		return VerifiedSnapshot{}, fmt.Errorf("snapshot outlives authority endpoints")
	}
	root, rid, _ := certificates.Parse(a.DER())
	keys := map[string]bool{string(root.RawSubjectPublicKeyInfo): true, string(c.RawSubjectPublicKeyInfo): true}
	caSerials := map[string]bool{rid.Serial: true, sid.Serial: true}
	last := ""
	for _, der := range p.Issuers {
		issuer, id, err := certificates.Verify(a, der, nil, certificates.Issuer, now)
		if err != nil {
			return VerifiedSnapshot{}, err
		}
		if id.Serial <= last || keys[string(issuer.RawSubjectPublicKeyInfo)] || p.ExpiresAt > issuer.NotAfter.Unix() {
			return VerifiedSnapshot{}, fmt.Errorf("duplicate issuer, reused CA key, or invalid issuer lifetime")
		}
		last = id.Serial
		keys[string(issuer.RawSubjectPublicKeyInfo)] = true
		caSerials[id.Serial] = true
	}
	for i, r := range p.EligibleRoles {
		if (r != Authority && r != Controller && r != Coordinator && r != Node && r != Worker) || (i > 0 && p.EligibleRoles[i-1] >= r) {
			return VerifiedSnapshot{}, fmt.Errorf("invalid or duplicate role eligibility")
		}
	}
	last = ""
	for _, r := range p.Revocations {
		key := r.Kind + ":" + r.ID
		if (r.Kind != "principal" && r.Kind != "certificate") || !certificates.ValidID(r.ID) || (r.Mode != "prospective" && r.Mode != "compromise") ||
			!validTextReason(r.Reason) || r.FirstGeneration == 0 || r.FirstGeneration > p.Generation || key <= last {
			return VerifiedSnapshot{}, fmt.Errorf("invalid or unordered revocation")
		}
		if r.Kind == "certificate" && caSerials[r.ID] {
			return VerifiedSnapshot{}, fmt.Errorf("CA compromise requires cluster rekey, not leaf revocation")
		}
		last = key
	}
	return VerifiedSnapshot{anchor: a, signed: s}, nil
}

func validTextReason(s string) bool {
	if len(s) == 0 || len(s) > 256 {
		return false
	}
	for _, c := range s {
		if c < 0x20 || c > 0x7e {
			return false
		}
	}
	return true
}

func DecodeSnapshot(a Anchor, body []byte, now time.Time) (VerifiedSnapshot, error) {
	var s SignedSnapshot
	if err := decodeCanonical(body, &s); err != nil {
		return VerifiedSnapshot{}, err
	}
	return VerifySnapshot(a, s, now)
}

// RestoreSnapshot validates previously persisted trust, including an expired
// snapshot, so Advance can refresh it without forgetting revocations. Call only
// on protected local state with an independently pinned root. The restored view
// still fails fresh checks for new operations until it is advanced.
func RestoreSnapshot(a Anchor, body []byte, now time.Time) (VerifiedSnapshot, error) {
	if _, err := certificates.Pin(a.DER(), a.Fingerprint(), now); err != nil {
		return VerifiedSnapshot{}, err
	}
	var s SignedSnapshot
	if err := decodeCanonical(body, &s); err != nil {
		return VerifiedSnapshot{}, err
	}
	if s.Payload.IssuedAt > now.Add(certificates.ClockSkew).Unix() {
		return VerifiedSnapshot{}, fmt.Errorf("persisted snapshot is from the future")
	}
	return VerifySnapshot(a, s, time.Unix(s.Payload.IssuedAt, 0))
}

// Advance supports stale-state recovery using only the pinned root and a fresh
// new signature. Revocation tombstones never disappear or move backwards.
func (v VerifiedSnapshot) Advance(next SignedSnapshot, now time.Time) (VerifiedSnapshot, error) {
	n, err := VerifySnapshot(v.anchor, next, now)
	if err != nil {
		return VerifiedSnapshot{}, err
	}
	if n.Generation() <= v.Generation() || n.signed.Payload.IssuedAt < v.signed.Payload.IssuedAt {
		return VerifiedSnapshot{}, fmt.Errorf("trust snapshot rollback or replay")
	}
	previousEndpoints, nextEndpoints := v.signed.Payload.Endpoints.Payload, n.signed.Payload.Endpoints.Payload
	previousBody, _ := json.Marshal(previousEndpoints)
	nextBody, _ := json.Marshal(nextEndpoints)
	if nextEndpoints.IssuedAt < previousEndpoints.IssuedAt || (nextEndpoints.IssuedAt == previousEndpoints.IssuedAt && !bytes.Equal(previousBody, nextBody)) {
		return VerifiedSnapshot{}, fmt.Errorf("authority endpoint rollback or same-time conflict")
	}
	remaining := make(map[string]Revocation, len(v.signed.Payload.Revocations))
	for _, old := range v.signed.Payload.Revocations {
		remaining[old.Kind+":"+old.ID] = old
	}
	for _, r := range n.signed.Payload.Revocations {
		key := r.Kind + ":" + r.ID
		if old, found := remaining[key]; found {
			if r.FirstGeneration != old.FirstGeneration || (old.Mode == "compromise" && r.Mode != "compromise") {
				return VerifiedSnapshot{}, fmt.Errorf("revocation rollback")
			}
			delete(remaining, key)
		} else if r.FirstGeneration <= v.Generation() {
			return VerifiedSnapshot{}, fmt.Errorf("new revocation backdates a known generation")
		}
	}
	if len(remaining) != 0 {
		return VerifiedSnapshot{}, fmt.Errorf("revocation tombstone removed")
	}
	return n, nil
}

func (v VerifiedSnapshot) fresh(now time.Time) error {
	return timeWindow(v.signed.Payload.IssuedAt, v.signed.Payload.ExpiresAt, SnapshotLifetime, now)
}

func (v VerifiedSnapshot) eligible(role Role) bool {
	for _, r := range v.signed.Payload.EligibleRoles {
		if r == role {
			return true
		}
	}
	return false
}

func objectDigest(v any) (string, error) {
	b, err := canonical(v)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}
