// Package ringtls implements only the workload stream TLS profile. The owning
// transport adapter supplies exact planned-peer and current-trust validation.
package ringtls

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/transport/internal/tlsidentity"
)

const Protocol = "mixlab-ring-v1"

type Identity = tlsidentity.Identity
type Verify func([][]byte, time.Time) error
type Policy struct {
	certificate tls.Certificate
	verify      Verify
	clock       func() time.Time
}

func New(local Identity, verify Verify, clock func() time.Time) (*Policy, error) {
	if verify == nil || clock == nil {
		return nil, fmt.Errorf("ring TLS requires a pinned workload verifier and clock")
	}
	c, err := tlsidentity.Certificate(local)
	if err != nil {
		return nil, err
	}
	return &Policy{c, verify, clock}, nil
}
func (p *Policy) VerifyConnection(s tls.ConnectionState) error {
	if s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != Protocol || len(s.PeerCertificates) != 3 {
		return fmt.Errorf("invalid ring TLS profile")
	}
	chain, err := tlsidentity.PeerChain(s)
	if err != nil {
		return err
	}
	return p.verify(chain, p.clock())
}
func (p *Policy) base() *tls.Config {
	c := p.certificate
	c.Certificate = make([][]byte, len(p.certificate.Certificate))
	for i, der := range p.certificate.Certificate {
		c.Certificate[i] = bytes.Clone(der)
	}
	c.Leaf = nil
	return &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13, Certificates: []tls.Certificate{c}, NextProtos: []string{Protocol}, SessionTicketsDisabled: true, Time: p.clock, VerifyConnection: p.VerifyConnection}
}
func (p *Policy) ServerConfig() *tls.Config {
	c := p.base()
	c.ClientAuth = tls.RequireAnyClientCert
	return c
}
func (p *Policy) ClientConfig() *tls.Config {
	c := p.base()
	c.InsecureSkipVerify = true /* mandatory pinned workload verifier */
	return c
}
