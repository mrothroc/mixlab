package enrollmenttls

import (
	"crypto/tls"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/transport/internal/tlsidentity"
)

type Identity = tlsidentity.Identity

// AcceptSource is a local cluster-trust/presentation port. It must pin and
// validate the source under the explicitly selected enrollment policy before
// returning nil. This adapter never implicitly accepts a proposed root.
type AcceptSource func(chain [][]byte, now time.Time) error

func profile(clock func() time.Time) (*tls.Config, error) {
	if clock == nil {
		return nil, fmt.Errorf("enrollment TLS clock required")
	}
	return &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13, NextProtos: []string{Protocol}, SessionTicketsDisabled: true, Time: clock}, nil
}

func ServerConfig(local Identity, clock func() time.Time) (*tls.Config, error) {
	c, err := profile(clock)
	if err != nil {
		return nil, err
	}
	identity, err := tlsidentity.Certificate(local)
	if err != nil {
		return nil, err
	}
	c.Certificates = []tls.Certificate{identity}
	c.ClientAuth = tls.NoClientCert
	c.VerifyConnection = func(s tls.ConnectionState) error {
		if s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != Protocol {
			return fmt.Errorf("invalid provisional enrollment TLS profile")
		}
		return nil
	}
	return c, nil
}

// ClientConfig is confined to enrollment. Unlike normal managed TLS, no client
// identity exists yet. The mandatory acceptance port authenticates the source
// under the selected policy; it must never be replaced by an accept-all callback.
func ClientConfig(accept AcceptSource, clock func() time.Time) (*tls.Config, error) {
	if accept == nil {
		return nil, fmt.Errorf("explicit enrollment source acceptance required")
	}
	c, err := profile(clock)
	if err != nil {
		return nil, err
	}
	c.InsecureSkipVerify = true // mandatory enrollment-policy verifier below
	c.VerifyConnection = func(s tls.ConnectionState) error {
		if s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != Protocol {
			return fmt.Errorf("invalid provisional enrollment TLS profile")
		}
		chain, err := tlsidentity.PeerChain(s)
		if err != nil {
			return err
		}
		return accept(chain, clock())
	}
	return c, nil
}
