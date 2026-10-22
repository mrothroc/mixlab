package clusterdiagnostic

import (
	"context"
	"net"
	"syscall"

	"github.com/mrothroc/mixlab/transport/managedtls"
)

// FromReason reconstructs only known, static public guidance from persisted
// classification. Never echo an arbitrary diagnostic string from disk/peers.
func FromReason(reason string) Failure {
	var sample error
	switch reason {
	case "local_identity_unavailable":
		sample = ErrLocalIdentity
	case "canceled":
		sample = context.Canceled
	case "route_or_permission_denied":
		sample = syscall.EHOSTUNREACH
	case "connection_refused":
		sample = syscall.ECONNREFUSED
	case "permission_denied":
		sample = syscall.EACCES
	case "dns_failed":
		sample = &net.DNSError{}
	case "timeout":
		sample = context.DeadlineExceeded
	case "tls_rejected":
		sample = managedtls.ErrPeerRejected
	}
	return Classify(sample)
}
