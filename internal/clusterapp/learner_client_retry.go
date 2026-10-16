package clusterapp

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"net"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

const (
	learnerRetryBudget   = 20 * time.Second
	learnerAttemptBudget = 5 * time.Second
)

func retryLearnerSemantic(parent context.Context, attempt func(context.Context) (learnerapi.Response, bool, error)) (learnerapi.Response, error) {
	ctx, cancel := context.WithTimeout(parent, learnerRetryBudget)
	defer cancel()
	delay := 100 * time.Millisecond
	for {
		if err := ctx.Err(); err != nil {
			return learnerapi.Response{}, err
		}
		call, stop := context.WithTimeout(ctx, learnerAttemptBudget)
		out, retry, err := attempt(call)
		stop()
		if err == nil {
			return out, ctx.Err()
		}
		if !retry || ctx.Err() != nil {
			return learnerapi.Response{}, errors.Join(err, ctx.Err())
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return learnerapi.Response{}, errors.Join(err, ctx.Err())
		case <-timer.C:
		}
		if delay < time.Second {
			delay *= 2
			if delay > time.Second {
				delay = time.Second
			}
		}
	}
}

// Classify transport failures only, never policy/HTTP/decoder errors. A peer
// rejection can wrap a timeout/EOF from a trust port and must still fail closed.
func transientLearnerTransport(err error) bool {
	var alert tls.AlertError
	var verify *tls.CertificateVerificationError
	var invalid x509.CertificateInvalidError
	var unknown x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var op *net.OpError
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, managedtls.ErrPeerRejected) ||
		errors.As(err, &alert) || errors.As(err, &verify) || errors.As(err, &invalid) ||
		errors.As(err, &unknown) || errors.As(err, &hostname) || (errors.As(err, &op) && op.Op == "remote error") {
		return false
	}
	if errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNABORTED) ||
		errors.Is(err, syscall.EPIPE) || errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var timeout net.Error
	return errors.As(err, &timeout) && timeout.Timeout()
}
