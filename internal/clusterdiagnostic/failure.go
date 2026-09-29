// Package clusterdiagnostic maps local observations to safe operator guidance.
// It never relaxes authentication or publishes remote error bodies/secret paths.
package clusterdiagnostic

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"net"
	"syscall"

	"github.com/mrothroc/mixlab/transport/managedtls"
)

type Failure struct {
	Reason string `json:"reason"`
	Hint   string `json:"hint"`
}

var ErrLocalIdentity = errors.New("local identity or trust unavailable")

func Classify(err error) Failure {
	var dns *net.DNSError
	var op *net.OpError
	var alert tls.AlertError
	var verify *tls.CertificateVerificationError
	var cert x509.CertificateInvalidError
	var timeout net.Error
	switch {
	case errors.Is(err, ErrLocalIdentity):
		return Failure{"local_identity_unavailable", "Run doctor with this principal to check local trust freshness, credential validity and Keychain access."}
	case errors.Is(err, context.Canceled):
		return Failure{"canceled", "The request was canceled."}
	case errors.Is(err, syscall.EHOSTUNREACH), errors.Is(err, syscall.ENETUNREACH):
		return Failure{"route_or_permission_denied", "Check routing and host policy. On macOS, repeated failures to an on-link peer can indicate Local Network denial; allow the signed service at the console. This error alone does not prove permission denial."}
	case errors.Is(err, syscall.ECONNREFUSED):
		return Failure{"connection_refused", "Check the listener address and service status on the target host."}
	case errors.Is(err, syscall.EACCES), errors.Is(err, syscall.EPERM):
		return Failure{"permission_denied", "Check local network policy and service permissions; do not disable the firewall."}
	case errors.As(err, &dns):
		return Failure{"dns_failed", "Check name resolution or use an explicit LAN address."}
	case errors.Is(err, context.DeadlineExceeded):
		return Failure{"timeout", "Check peer availability, routing and firewall rules; a timeout does not identify which one failed."}
	case errors.Is(err, managedtls.ErrPeerRejected), errors.As(err, &alert), errors.As(err, &verify), errors.As(err, &cert):
		return Failure{"tls_rejected", "Check both hosts' clock, enrollment and trust freshness. Remote TLS rejection does not prove which credential or trust state failed."}
	case errors.As(err, &op) && op.Op == "remote error":
		return Failure{"tls_rejected", "Inspect the peer service log for stale trust, expiry or rejection; no authentication checks were bypassed."}
	case errors.As(err, &timeout) && timeout.Timeout():
		return Failure{"timeout", "Check peer availability and network policy."}
	default:
		return Failure{"unavailable_or_unauthenticated", "Inspect local doctor output and peer service logs; the cause has not been established."}
	}
}
