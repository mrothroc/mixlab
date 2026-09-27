package ringtls

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"testing"
	"time"
)

// chain builds the three-certificate identity tlsidentity.Certificate demands.
func chain(t *testing.T) Identity {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "ring-verifier-test"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		IsCA:         true,
		KeyUsage:     x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, public, private)
	if err != nil {
		t.Fatal(err)
	}
	return Identity{Chain: [][]byte{der, der, der}, Key: private}
}

// Every config here sets InsecureSkipVerify, which switches OFF Go's built-in
// peer verification. That is only safe because VerifyConnection replaces it.
// Dropping the VerifyConnection assignment compiles cleanly and disables peer
// verification outright, and nothing else in the tree notices -- verified by
// deleting it and running the suite, which stayed green. This pins the pairing
// so the two can never be separated silently.
func TestConfigsNeverSkipVerificationWithoutAReplacement(t *testing.T) {
	p, err := New(chain(t), func([][]byte, time.Time) error { return nil }, time.Now)
	if err != nil {
		t.Fatal(err)
	}
	for name, config := range map[string]*tls.Config{"client": p.ClientConfig(), "server": p.ServerConfig()} {
		if config.InsecureSkipVerify && config.VerifyConnection == nil {
			t.Fatalf("%s config skips Go's verification with no replacement verifier", name)
		}
		if config.VerifyConnection == nil {
			t.Fatalf("%s config has no peer verifier", name)
		}
	}
}

// The verifier must actually reject: a wired-but-permissive callback would
// satisfy the nil check above while verifying nothing.
func TestVerifierRejectsAnUnacceptablePeer(t *testing.T) {
	p, err := New(chain(t), func([][]byte, time.Time) error { return nil }, time.Now)
	if err != nil {
		t.Fatal(err)
	}
	// A connection state that fails the profile must be refused regardless of
	// what the injected workload verifier says.
	if err := p.ClientConfig().VerifyConnection(tls.ConnectionState{}); err == nil {
		t.Fatal("verifier accepted an empty connection state")
	}
	if err := p.ClientConfig().VerifyConnection(tls.ConnectionState{
		Version: tls.VersionTLS13, NegotiatedProtocol: Protocol,
	}); err == nil {
		t.Fatal("verifier accepted a peer presenting no certificates")
	}
}

// New must refuse to build a policy with no verifier, since every config it
// returns disables Go's own verification.
func TestNewRequiresAVerifier(t *testing.T) {
	if _, err := New(chain(t), nil, time.Now); err == nil {
		t.Fatal("built a ring TLS policy with no pinned verifier")
	}
}
