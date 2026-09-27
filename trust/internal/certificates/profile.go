// Package certificates implements the v1 trust certificate adapter using the
// runtime X.509/Ed25519 implementation. Only the trust context may issue through
// this package; enrollment authorization is not a certificate-adapter concern.
package certificates

import (
	"bytes"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/hex"
	"fmt"
	"math/big"
	"net/url"
	"strings"
	"time"
)

type Profile string
type Role string

const (
	Root                Profile = "root"
	Issuer              Profile = "issuer"
	SnapshotSigner      Profile = "snapshot-signer"
	Principal           Profile = "principal"
	Authority           Role    = "authority"
	Controller          Role    = "controller"
	Coordinator         Role    = "coordinator"
	ClockSkew                   = 5 * time.Minute
	RootLifetime                = 3650 * 24 * time.Hour
	IssuerLifetime              = 365 * 24 * time.Hour
	PrincipalLifetime           = 30 * 24 * time.Hour
	MaxCertificateBytes         = 16 << 10
)

type Identity struct {
	Profile     Profile
	Cluster     string
	Role        Role
	Principal   string
	Serial      string
	EnvelopeKey []byte
	Workload    *WorkloadBinding
}

// Anchor is an immutable pinned root, never a system-root pool or an address.
type Anchor struct {
	der                  []byte
	fingerprint, cluster string
}

func (a Anchor) DER() []byte         { return bytes.Clone(a.der) }
func (a Anchor) Fingerprint() string { return a.fingerprint }
func (a Anchor) Cluster() string     { return a.cluster }

func NewID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func ValidID(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 16 && hex.EncodeToString(b) == s
}

func Fingerprint(public crypto.PublicKey) (string, error) {
	key, ok := public.(ed25519.PublicKey)
	if !ok || len(key) != ed25519.PublicKeySize {
		return "", fmt.Errorf("trust v1 requires Ed25519")
	}
	der, err := x509.MarshalPKIXPublicKey(key)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(der)
	return hex.EncodeToString(h[:]), nil
}

func Pin(der []byte, fingerprint string, now time.Time) (Anchor, error) {
	c, id, err := Parse(der)
	if err != nil {
		return Anchor{}, err
	}
	got, err := Fingerprint(c.PublicKey)
	if err != nil || id.Profile != Root || fingerprint != got {
		return Anchor{}, fmt.Errorf("root pin/profile mismatch")
	}
	if !bytes.Equal(c.RawIssuer, c.RawSubject) || c.CheckSignatureFrom(c) != nil {
		return Anchor{}, fmt.Errorf("root is not self signed")
	}
	if err := live(c, now); err != nil {
		return Anchor{}, err
	}
	return Anchor{der: bytes.Clone(der), fingerprint: got, cluster: id.Cluster}, nil
}

func supportedRole(role Role) bool {
	return role == Authority || role == Controller || role == Coordinator || role == Node || role == Worker
}

// Parse checks the complete supported profile, not merely a URI or common name.
func Parse(der []byte) (*x509.Certificate, Identity, error) {
	var id Identity
	if len(der) == 0 || len(der) > MaxCertificateBytes {
		return nil, id, fmt.Errorf("certificate size out of bounds")
	}
	c, err := x509.ParseCertificate(der)
	if err != nil {
		return nil, id, fmt.Errorf("parse trust certificate: %w", err)
	}
	if _, err := Fingerprint(c.PublicKey); err != nil {
		return nil, id, err
	}
	if c.SignatureAlgorithm != x509.PureEd25519 || !c.BasicConstraintsValid || len(c.URIs) < 1 || len(c.URIs) > 2 ||
		len(c.DNSNames) != 0 || len(c.IPAddresses) != 0 || len(c.EmailAddresses) != 0 || len(c.UnknownExtKeyUsage) != 0 ||
		len(c.Subject.Names) != 0 || c.SerialNumber.Sign() <= 0 || c.SerialNumber.BitLen() > 128 || c.SerialNumber.BitLen() < 121 {
		return nil, id, fmt.Errorf("invalid trust certificate profile")
	}
	parts := strings.Split(c.URIs[0].String(), ":")
	if len(parts) != 6 || parts[0] != "urn" || parts[1] != "mixlab" || !ValidID(parts[3]) {
		return nil, id, fmt.Errorf("invalid canonical trust URI")
	}
	id.Cluster = parts[3]
	id.Serial = fmt.Sprintf("%032x", c.SerialNumber)
	id.Profile = Profile(parts[4])
	lifetime := IssuerLifetime
	switch parts[2] {
	case "principal":
		id.Profile = Principal
		id.Role = Role(parts[4])
		id.Principal = parts[5]
		if !supportedRole(id.Role) || !ValidID(id.Principal) || c.IsCA || c.KeyUsage != x509.KeyUsageDigitalSignature ||
			len(c.ExtKeyUsage) != 2 || c.ExtKeyUsage[0] != x509.ExtKeyUsageClientAuth || c.ExtKeyUsage[1] != x509.ExtKeyUsageServerAuth {
			return nil, id, fmt.Errorf("unsupported principal role or key usage")
		}
		lifetime = PrincipalLifetime
		if id.Role == Worker {
			lifetime = WorkloadLifetime
		}
	case "trust":
		fingerprint, _ := Fingerprint(c.PublicKey)
		if parts[5] != fingerprint || len(c.ExtKeyUsage) != 0 {
			return nil, id, fmt.Errorf("invalid trust signing identity")
		}
		switch id.Profile {
		case Root:
			lifetime = RootLifetime
			if !c.IsCA || c.MaxPathLen != 1 || c.MaxPathLenZero || c.KeyUsage != x509.KeyUsageCertSign {
				return nil, id, fmt.Errorf("invalid root constraints")
			}
		case Issuer:
			if !c.IsCA || c.MaxPathLen != 0 || !c.MaxPathLenZero || c.KeyUsage != x509.KeyUsageCertSign {
				return nil, id, fmt.Errorf("invalid issuer constraints")
			}
		case SnapshotSigner:
			if c.IsCA || c.KeyUsage != x509.KeyUsageDigitalSignature {
				return nil, id, fmt.Errorf("invalid snapshot-signer constraints")
			}
		default:
			return nil, id, fmt.Errorf("unknown trust certificate profile")
		}
	default:
		return nil, id, fmt.Errorf("unknown trust URI namespace")
	}
	if c.NotAfter.Sub(c.NotBefore) <= 0 || c.NotAfter.Sub(c.NotBefore) > lifetime+ClockSkew {
		return nil, id, fmt.Errorf("invalid profile lifetime")
	}
	if err := parseBindings(c, &id); err != nil {
		return nil, id, err
	}
	return c, id, nil
}

func live(c *x509.Certificate, now time.Time) error {
	if now.Before(c.NotBefore) || !now.Before(c.NotAfter) {
		return fmt.Errorf("trust certificate not currently valid")
	}
	return nil
}

// Verify requires a pinned root and exact two- or three-certificate hierarchy.
// The explicit pool prevents accidental platform-root or AIA trust fallback.
func Verify(a Anchor, der, issuerDER []byte, profile Profile, now time.Time) (*x509.Certificate, Identity, error) {
	c, id, err := Parse(der)
	if err != nil {
		return nil, id, err
	}
	root, rootID, err := Parse(a.der)
	if err != nil || rootID.Profile != Root || id.Cluster != a.cluster || id.Profile != profile {
		return nil, id, fmt.Errorf("wrong cluster or certificate purpose")
	}
	roots := x509.NewCertPool()
	roots.AddCert(root)
	intermediates := x509.NewCertPool()
	parent := root
	if profile == Principal {
		parent, _, err = Verify(a, issuerDER, nil, Issuer, now)
		if err != nil {
			return nil, id, err
		}
		intermediates.AddCert(parent)
	} else if len(issuerDER) != 0 || (profile != Issuer && profile != SnapshotSigner) {
		return nil, id, fmt.Errorf("invalid trust chain shape")
	}
	if bytes.Equal(c.RawSubjectPublicKeyInfo, root.RawSubjectPublicKeyInfo) || bytes.Equal(c.RawSubjectPublicKeyInfo, parent.RawSubjectPublicKeyInfo) {
		return nil, id, fmt.Errorf("CA and subordinate keys must be distinct")
	}
	if c.NotBefore.Before(parent.NotBefore) || c.NotAfter.After(parent.NotAfter) {
		return nil, id, fmt.Errorf("child lifetime exceeds issuer")
	}
	if err = live(c, now); err != nil {
		return nil, id, err
	}
	if err = live(root, now); err != nil {
		return nil, id, err
	}
	if err = c.CheckSignatureFrom(parent); err != nil {
		return nil, id, fmt.Errorf("certificate not signed by its required immediate issuer: %w", err)
	}
	if _, err = c.Verify(x509.VerifyOptions{Roots: roots, Intermediates: intermediates, CurrentTime: now, KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageAny}}); err != nil {
		return nil, id, fmt.Errorf("trust chain verification: %w", err)
	}
	return c, id, nil
}

func MatchSigner(c *x509.Certificate, key crypto.Signer) error {
	if key == nil {
		return fmt.Errorf("missing signer handle")
	}
	fp, err := Fingerprint(key.Public())
	want, e := Fingerprint(c.PublicKey)
	if err != nil || e != nil || fp != want {
		return fmt.Errorf("signer handle does not match certificate")
	}
	return nil
}

func CreateRoot(cluster string, key crypto.Signer, now time.Time) ([]byte, error) {
	if !ValidID(cluster) || key == nil {
		return nil, fmt.Errorf("invalid root inputs")
	}
	c, err := template(Root, cluster, "", "", key.Public(), now)
	if err != nil {
		return nil, err
	}
	return x509.CreateCertificate(rand.Reader, c, c, key.Public(), key)
}

// Issue is internal to cluster trust. Enrollment/workload policy must approve
// before invoking it. There is no issuance route in the public CLI yet.
func Issue(a Anchor, parentDER []byte, parentKey crypto.Signer, profile Profile, role Role, principal string, public crypto.PublicKey, now time.Time) ([]byte, error) {
	if role == Node || role == Worker {
		return nil, fmt.Errorf("bound roles require dedicated issuance")
	}
	parent, parentID, err := Parse(parentDER)
	if err != nil {
		return nil, err
	}
	if parentID.Cluster != a.cluster {
		return nil, fmt.Errorf("wrong issuing cluster")
	}
	if profile == Principal {
		if _, _, err = Verify(a, parentDER, nil, Issuer, now); err != nil {
			return nil, err
		}
	} else if (profile != Issuer && profile != SnapshotSigner) || !bytes.Equal(parentDER, a.der) {
		return nil, fmt.Errorf("wrong issuing purpose")
	}
	if err = live(parent, now); err != nil {
		return nil, err
	}
	if err = MatchSigner(parent, parentKey); err != nil {
		return nil, err
	}
	c, err := template(profile, a.cluster, role, principal, public, now)
	if err != nil {
		return nil, err
	}
	pub, _ := Fingerprint(public)
	parentPub, _ := Fingerprint(parent.PublicKey)
	if pub == parentPub || pub == a.fingerprint {
		return nil, fmt.Errorf("CA and subordinate keys must be distinct")
	}
	if c.NotBefore.Before(parent.NotBefore) {
		c.NotBefore = parent.NotBefore
	}
	if c.NotAfter.After(parent.NotAfter) {
		c.NotAfter = parent.NotAfter
	}
	return x509.CreateCertificate(rand.Reader, c, parent, public, parentKey)
}

func template(profile Profile, cluster string, role Role, principal string, public crypto.PublicKey, now time.Time) (*x509.Certificate, error) {
	fp, err := Fingerprint(public)
	if err != nil {
		return nil, err
	}
	serial := make([]byte, 16)
	if _, err = rand.Read(serial); err != nil {
		return nil, err
	}
	serial[0] |= 0x80 // fixed 128-bit positive integer; ASN.1 adds the sign byte
	c := &x509.Certificate{SerialNumber: new(big.Int).SetBytes(serial), SignatureAlgorithm: x509.PureEd25519, BasicConstraintsValid: true,
		NotBefore: now.UTC().Truncate(time.Second).Add(-ClockSkew), MaxPathLen: -1}
	uri := "urn:mixlab:trust:" + cluster + ":" + string(profile) + ":" + fp
	lifetime := IssuerLifetime
	switch profile {
	case Root:
		c.IsCA = true
		c.KeyUsage = x509.KeyUsageCertSign
		c.MaxPathLen = 1
		lifetime = RootLifetime
	case Issuer:
		c.IsCA = true
		c.KeyUsage = x509.KeyUsageCertSign
		c.MaxPathLen = 0
		c.MaxPathLenZero = true
	case SnapshotSigner:
		c.KeyUsage = x509.KeyUsageDigitalSignature
	case Principal:
		if !supportedRole(role) || !ValidID(principal) {
			return nil, fmt.Errorf("unsupported principal role/profile")
		}
		c.KeyUsage = x509.KeyUsageDigitalSignature
		c.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth}
		lifetime = PrincipalLifetime
		uri = "urn:mixlab:principal:" + cluster + ":" + string(role) + ":" + principal
	default:
		return nil, fmt.Errorf("unknown certificate profile")
	}
	if profile != Principal && (role != "" || principal != "") {
		return nil, fmt.Errorf("CA-purpose identity cannot carry an application role")
	}
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	c.URIs = []*url.URL{u}
	c.NotAfter = now.UTC().Truncate(time.Second).Add(lifetime)
	return c, nil
}
