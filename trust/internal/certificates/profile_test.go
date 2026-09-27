package certificates

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/hex"
	"math/big"
	"net/url"
	"strings"
	"testing"
	"time"
)

var testNow = time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)

func must[T any](t *testing.T, value T, err error) T {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func key(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	_, k, err := ed25519.GenerateKey(rand.Reader)
	return must(t, k, err)
}

func id(t *testing.T) string {
	t.Helper()
	v, err := NewID()
	return must(t, v, err)
}

type fixture struct {
	anchor                      Anchor
	root, issuer, leaf          []byte
	rootKey, issuerKey, leafKey ed25519.PrivateKey
}

func setup(t *testing.T) fixture {
	t.Helper()
	f := fixture{rootKey: key(t), issuerKey: key(t), leafKey: key(t)}
	var err error
	f.root, err = CreateRoot(id(t), f.rootKey, testNow)
	must(t, f.root, err)
	fp, err := Fingerprint(f.rootKey.Public())
	must(t, fp, err)
	f.anchor, err = Pin(f.root, fp, testNow)
	must(t, f.anchor, err)
	f.issuer, err = Issue(f.anchor, f.root, f.rootKey, Issuer, "", "", f.issuerKey.Public(), testNow)
	must(t, f.issuer, err)
	f.leaf, err = Issue(f.anchor, f.issuer, f.issuerKey, Principal, Controller, id(t), f.leafKey.Public(), testNow)
	must(t, f.leaf, err)
	return f
}

func TestProfilesAndPinnedHierarchy(t *testing.T) {
	f := setup(t)
	for _, role := range []Role{Authority, Controller, Coordinator} {
		t.Run(string(role), func(t *testing.T) {
			der, err := Issue(f.anchor, f.issuer, f.issuerKey, Principal, role, id(t), key(t).Public(), testNow)
			must(t, der, err)
			c, got, err := Verify(f.anchor, der, f.issuer, Principal, testNow)
			must(t, c, err)
			if got.Role != role || got.Cluster != f.anchor.Cluster() || !ValidID(got.Serial) {
				t.Fatalf("wrong identity: %+v", got)
			}
			if _, _, err := Verify(f.anchor, der, f.issuer, Principal, c.NotAfter); err == nil {
				t.Fatal("accepted at exclusive expiry")
			}
		})
	}
	signer, err := Issue(f.anchor, f.root, f.rootKey, SnapshotSigner, "", "", key(t).Public(), testNow)
	must(t, signer, err)
	if _, _, err := Verify(f.anchor, signer, nil, SnapshotSigner, testNow); err != nil {
		t.Fatal(err)
	}
	other := setup(t)
	for name, check := range map[string]func() error{
		"wrong root":       func() error { _, _, e := Verify(other.anchor, f.leaf, f.issuer, Principal, testNow); return e },
		"wrong issuer":     func() error { _, _, e := Verify(f.anchor, f.leaf, other.issuer, Principal, testNow); return e },
		"no issuer":        func() error { _, _, e := Verify(f.anchor, f.leaf, nil, Principal, testNow); return e },
		"wrong purpose":    func() error { _, _, e := Verify(f.anchor, signer, nil, Issuer, testNow); return e },
		"missing pin":      func() error { _, _, e := Verify(Anchor{}, f.leaf, f.issuer, Principal, testNow); return e },
		"wrong pin":        func() error { _, e := Pin(f.root, strings.Repeat("0", 64), testNow); return e },
		"leaf as root":     func() error { fp, _ := Fingerprint(f.leafKey.Public()); _, e := Pin(f.leaf, fp, testNow); return e },
		"wrong key handle": func() error { c, _, _ := Parse(f.leaf); return MatchSigner(c, f.rootKey) },
		"no key handle":    func() error { c, _, _ := Parse(f.leaf); return MatchSigner(c, nil) },
	} {
		t.Run(name, func(t *testing.T) {
			if check() == nil {
				t.Fatal("accepted invalid trust hierarchy")
			}
		})
	}
	copyDER := f.anchor.DER()
	copyDER[0] ^= 1
	f.root[0] ^= 1
	if _, err := Pin(f.anchor.DER(), f.anchor.Fingerprint(), testNow); err != nil {
		t.Fatalf("anchor aliases caller bytes: %v", err)
	}
}

func TestProfileRejectsMalformedConstraints(t *testing.T) {
	f := setup(t)
	parent, _, _ := Parse(f.issuer)
	for name, mutate := range map[string]func(*x509.Certificate){
		"subject identity": func(c *x509.Certificate) { c.Subject.CommonName = "controller" },
		"multiple URIs":    func(c *x509.Certificate) { c.URIs = append(c.URIs, c.URIs[0]) },
		"DNS identity":     func(c *x509.Certificate) { c.DNSNames = []string{"localhost"} },
		"wrong namespace": func(c *x509.Certificate) {
			u, _ := url.Parse("urn:other:principal:" + f.anchor.Cluster() + ":controller:" + id(t))
			c.URIs = []*url.URL{u}
		},
		"unknown critical extension": func(c *x509.Certificate) {
			c.ExtraExtensions = []pkix.Extension{{Id: asn1.ObjectIdentifier{1, 2, 3, 4}, Critical: true, Value: []byte{5, 0}}}
		},
		"CA leaf":             func(c *x509.Certificate) { c.IsCA = true },
		"cert sign usage":     func(c *x509.Certificate) { c.KeyUsage |= x509.KeyUsageCertSign },
		"extra EKU":           func(c *x509.Certificate) { c.ExtKeyUsage = append(c.ExtKeyUsage, x509.ExtKeyUsageCodeSigning) },
		"missing EKU":         func(c *x509.Certificate) { c.ExtKeyUsage = nil },
		"missing constraints": func(c *x509.Certificate) { c.BasicConstraintsValid = false },
		"long lifetime":       func(c *x509.Certificate) { c.NotAfter = c.NotAfter.Add(time.Second) },
		"short serial":        func(c *x509.Certificate) { c.SerialNumber = big.NewInt(1) },
		"oversized serial":    func(c *x509.Certificate) { c.SerialNumber = new(big.Int).Lsh(big.NewInt(1), 128) },
	} {
		t.Run(name, func(t *testing.T) {
			c, err := template(Principal, f.anchor.Cluster(), Controller, id(t), f.leafKey.Public(), testNow)
			must(t, c, err)
			mutate(c)
			der, err := x509.CreateCertificate(rand.Reader, c, parent, f.leafKey.Public(), f.issuerKey)
			must(t, der, err)
			if _, _, err := Parse(der); err == nil {
				t.Fatal("accepted malformed profile")
			}
		})
	}
	for _, der := range [][]byte{nil, {0}, make([]byte, MaxCertificateBytes+1)} {
		if _, _, err := Parse(der); err == nil {
			t.Fatal("accepted malformed DER")
		}
	}
}

func TestIssueRejectsRoleAndKeyConfusion(t *testing.T) {
	f := setup(t)
	for name, check := range map[string]func() error{
		"principal via root": func() error {
			_, e := Issue(f.anchor, f.root, f.rootKey, Principal, Controller, id(t), f.leafKey.Public(), testNow)
			return e
		},
		"CA via issuer": func() error {
			_, e := Issue(f.anchor, f.issuer, f.issuerKey, Issuer, "", "", key(t).Public(), testNow)
			return e
		},
		"CA role": func() error {
			_, e := Issue(f.anchor, f.root, f.rootKey, Issuer, Controller, id(t), key(t).Public(), testNow)
			return e
		},
		"CA reused key": func() error {
			_, e := Issue(f.anchor, f.root, f.rootKey, Issuer, "", "", f.rootKey.Public(), testNow)
			return e
		},
		"leaf reused key": func() error {
			_, e := Issue(f.anchor, f.issuer, f.issuerKey, Principal, Controller, id(t), f.issuerKey.Public(), testNow)
			return e
		},
		"node missing binding": func() error {
			_, e := Issue(f.anchor, f.issuer, f.issuerKey, Principal, "node", id(t), f.leafKey.Public(), testNow)
			return e
		},
		"worker missing binding": func() error {
			_, e := Issue(f.anchor, f.issuer, f.issuerKey, Principal, "worker", id(t), f.leafKey.Public(), testNow)
			return e
		},
		"expired issuer": func() error {
			_, e := Issue(f.anchor, f.issuer, f.issuerKey, Principal, Controller, id(t), f.leafKey.Public(), testNow.Add(IssuerLifetime))
			return e
		},
	} {
		t.Run(name, func(t *testing.T) {
			if check() == nil {
				t.Fatal("accepted invalid issuance")
			}
		})
	}
	// A generic X.509 verifier could otherwise select the root and silently
	// ignore the supplied intermediate, bypassing the exact hierarchy policy.
	c, err := template(Principal, f.anchor.Cluster(), Controller, id(t), f.leafKey.Public(), testNow)
	must(t, c, err)
	root, _, _ := Parse(f.root)
	der, err := x509.CreateCertificate(rand.Reader, c, root, f.leafKey.Public(), f.rootKey)
	must(t, der, err)
	if _, _, err := Verify(f.anchor, der, f.issuer, Principal, testNow); err == nil {
		t.Fatal("accepted root-direct principal")
	}
}

func TestFingerprintIsCanonicalSPKI(t *testing.T) {
	pub := ed25519.NewKeyFromSeed(make([]byte, 32)).Public().(ed25519.PublicKey)
	fp, err := Fingerprint(pub)
	must(t, fp, err)
	// RFC 8410 SubjectPublicKeyInfo prefix followed by the raw 32-byte key.
	spki := append([]byte{0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00}, pub...)
	h := sha256.Sum256(spki)
	if fp != hex.EncodeToString(h[:]) {
		t.Fatal("fingerprint not SPKI SHA-256")
	}
	for _, invalid := range []any{nil, []byte(pub), ed25519.PublicKey{1}} {
		if _, err := Fingerprint(invalid); err == nil {
			t.Fatal("accepted non-Ed25519 key")
		}
	}
	if ValidID(strings.Repeat("A", 32)) || ValidID("00") {
		t.Fatal("accepted noncanonical ID")
	}
	a, b := id(t), id(t)
	if a == b || !ValidID(a) {
		t.Fatal("IDs not independently generated")
	}
}

func TestIssueClipsLifetimeToParent(t *testing.T) {
	f := setup(t)
	now := testNow.Add(IssuerLifetime - time.Hour)
	der, err := Issue(f.anchor, f.issuer, f.issuerKey, Principal, Controller, id(t), f.leafKey.Public(), now)
	must(t, der, err)
	c, _, err := Verify(f.anchor, der, f.issuer, Principal, now)
	must(t, c, err)
	issuer, _, _ := Parse(f.issuer)
	if !c.NotAfter.Equal(issuer.NotAfter) {
		t.Fatal("leaf outlives issuer")
	}
	if bytes.Equal(c.RawSubjectPublicKeyInfo, issuer.RawSubjectPublicKeyInfo) {
		t.Fatal("key reuse")
	}
}

func FuzzParseCertificate(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0x30, 0x00})
	k := ed25519.NewKeyFromSeed(make([]byte, ed25519.SeedSize))
	der, err := CreateRoot(strings.Repeat("01", 16), k, testNow)
	if err != nil {
		f.Fatal(err)
	}
	f.Add(der)
	f.Fuzz(func(t *testing.T, der []byte) { _, _, _ = Parse(der) })
}
