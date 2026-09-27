// Package managedtls implements the normal managed TLS profile. It accepts
// certificate verification through a trust port; it owns no keys, enrollment
// decisions, application permissions, listeners, or persistent state.
package managedtls

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/transport/internal/tlsidentity"
	"github.com/mrothroc/mixlab/trust"
)

const Protocol = "http/1.1"

// Verify must use the owning context's current pinned trust and required peer
// identity. A successful result proves identity, never application authority.
// It can be called concurrently and must not retain or mutate the supplied DER.
type Verify func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error)

type Identity = tlsidentity.Identity

// Policy freezes public credentials while retaining the protected signer and
// current-trust port. It does not make a captured snapshot fresh automatically.
type Policy struct {
	certificate tls.Certificate
	verify      Verify
	clock       func() time.Time
}

// RequirePeer narrows an existing pinned policy before the HTTP body is sent.
// A successful response check alone is too late for mutating node operations.
func (p *Policy) RequirePeer(role trust.Role, id string) (*Policy, error) {
	if p == nil || id == "" || (role != trust.Node && role != trust.Controller && role != trust.Authority && role != trust.Coordinator) {
		return nil, fmt.Errorf("exact managed peer required")
	}
	narrowed := *p
	narrowed.verify = func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		peer, err := p.verify(chain, now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		if peer.Role != role || peer.Principal != id {
			return trust.AuthenticatedPrincipal{}, fmt.Errorf("managed peer differs from selected principal")
		}
		return peer, nil
	}
	return &narrowed, nil
}

func New(local Identity, verify Verify, clock func() time.Time) (*Policy, error) {
	if local.Key == nil || verify == nil || clock == nil || len(local.Chain) != 3 {
		return nil, fmt.Errorf("managed TLS requires a principal chain, protected signer, verifier, and clock")
	}
	certificate, err := tlsidentity.Certificate(local)
	if err != nil {
		return nil, err
	}
	return &Policy{certificate: certificate, verify: verify, clock: clock}, nil
}

// Authenticate rechecks current trust for an already-established connection.
// Call before each control request, not just on handshake: HTTP keepalive must
// not bypass revocation or expiry. A long-lived workload stream instead follows
// its owning job's admitted-channel and revocation/cleanup policy.
func (p *Policy) Authenticate(state tls.ConnectionState) (trust.AuthenticatedPrincipal, error) {
	if !state.HandshakeComplete {
		return trust.AuthenticatedPrincipal{}, fmt.Errorf("managed TLS handshake incomplete")
	}
	return p.peer(state)
}

func (p *Policy) peer(state tls.ConnectionState) (trust.AuthenticatedPrincipal, error) {
	if p == nil || state.Version != tls.VersionTLS13 || state.DidResume || state.NegotiatedProtocol != Protocol || len(state.PeerCertificates) != 3 {
		return trust.AuthenticatedPrincipal{}, fmt.Errorf("invalid managed TLS connection profile")
	}
	chain, err := tlsidentity.PeerChain(state)
	if err != nil {
		return trust.AuthenticatedPrincipal{}, err
	}
	return p.verify(chain, p.clock())
}

func (p *Policy) base() *tls.Config {
	// Each caller receives independent public slices. No session tickets/cache:
	// every normal managed connection performs a full key-possession handshake.
	c := p.certificate
	c.Certificate = make([][]byte, len(p.certificate.Certificate))
	for i, der := range p.certificate.Certificate {
		c.Certificate[i] = bytes.Clone(der)
	}
	c.Leaf = nil
	return &tls.Config{
		MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13,
		Certificates: []tls.Certificate{c}, NextProtos: []string{Protocol},
		SessionTicketsDisabled: true, Time: p.clock,
		VerifyConnection: func(s tls.ConnectionState) error { _, err := p.peer(s); return err },
	}
}

// ClientConfig uses the cluster's URI identities rather than Web PKI hostnames.
// InsecureSkipVerify disables ONLY Go's default Web PKI verifier; the mandatory
// VerifyConnection callback performs pinned-root/profile/current-trust checks.
// Composition must not replace callbacks or relax the returned configuration.
func (p *Policy) ClientConfig() *tls.Config {
	c := p.base()
	c.InsecureSkipVerify = true // mandatory custom cluster verifier above
	return c
}

func (p *Policy) ServerConfig() *tls.Config {
	c := p.base()
	c.ClientAuth = tls.RequireAnyClientCert // custom cluster verification above
	return c
}

// SnapshotServerConfig permits an anonymous TLS peer for public signed snapshot
// refresh only. A composition must put every other route behind AuthenticateHTTP
// (which still requires a current, exact principal chain). Supplied certificates
// are never ignored or downgraded to anonymous on verification failure.
func (p *Policy) SnapshotServerConfig() *tls.Config {
	c := p.base()
	c.ClientAuth = tls.RequestClientCert
	c.VerifyConnection = func(s tls.ConnectionState) error {
		if s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != Protocol {
			return fmt.Errorf("invalid trust-refresh TLS profile")
		}
		if len(s.PeerCertificates) == 0 {
			return nil
		}
		_, err := p.peer(s)
		return err
	}
	return c
}
