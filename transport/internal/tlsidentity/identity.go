// Package tlsidentity assembles a protected local TLS identity. Certificate
// profile and authorization decisions remain with the owning trust context.
package tlsidentity

import (
	"bytes"
	"crypto"
	"crypto/tls"
	"crypto/x509"
	"fmt"
)

const MaxCertificateBytes = 16 << 10

type Identity struct {
	Chain [][]byte
	Key   crypto.Signer
}

func Certificate(local Identity) (tls.Certificate, error) {
	if local.Key == nil || len(local.Chain) != 3 {
		return tls.Certificate{}, fmt.Errorf("TLS principal chain and protected signer required")
	}
	chain := make([][]byte, len(local.Chain))
	for i, der := range local.Chain {
		if len(der) == 0 || len(der) > MaxCertificateBytes {
			return tls.Certificate{}, fmt.Errorf("invalid TLS certificate size")
		}
		if _, err := x509.ParseCertificate(der); err != nil {
			return tls.Certificate{}, fmt.Errorf("invalid TLS certificate: %w", err)
		}
		chain[i] = bytes.Clone(der)
	}
	leaf, err := x509.ParseCertificate(chain[0])
	if err != nil {
		return tls.Certificate{}, err
	}
	want, err := x509.MarshalPKIXPublicKey(local.Key.Public())
	if err != nil || !bytes.Equal(want, leaf.RawSubjectPublicKeyInfo) {
		return tls.Certificate{}, fmt.Errorf("TLS signer does not match leaf certificate")
	}
	return tls.Certificate{Certificate: chain, PrivateKey: local.Key, Leaf: leaf}, nil
}

func PeerChain(state tls.ConnectionState) ([][]byte, error) {
	if len(state.PeerCertificates) != 3 {
		return nil, fmt.Errorf("TLS principal chain must have three certificates")
	}
	chain := make([][]byte, 3)
	for i, c := range state.PeerCertificates {
		if c == nil || len(c.Raw) == 0 || len(c.Raw) > MaxCertificateBytes {
			return nil, fmt.Errorf("invalid TLS peer chain")
		}
		chain[i] = bytes.Clone(c.Raw)
	}
	return chain, nil
}
