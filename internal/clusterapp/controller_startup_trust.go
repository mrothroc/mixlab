package clusterapp

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust/principal"
)

const (
	controllerStartupTrustBudget  = 2 * time.Minute
	controllerStartupTrustAttempt = remoteTrustRefreshTimeout
)

// RefreshControllerStartup retries only the pinned, read-only snapshot fetch.
// After transient exhaustion a currently valid cache can still authorize an
// offline cancellation. Authentication and signed-state errors never fall back.
func RefreshControllerStartup(ctx context.Context, p *principal.Store, endpoint string, clock func() time.Time, warning func(error)) error {
	if p == nil || clock == nil {
		return fmt.Errorf("controller principal and clock required")
	}
	return refreshControllerStartup(ctx, func(call context.Context) error {
		// The shared refresh reloads the local view and checks the selected
		// endpoint against that view on every attempt, including after waits.
		// Startup has its own bounded cache fallback, so use a single fetch
		// here rather than nesting the service refresh retry budget.
		return refreshRemoteTrustWithTimeout(call, p, endpoint, clock, controllerStartupTrustAttempt)
	}, func() error {
		_, _, err := p.Active(clock())
		return err
	}, warning)
}

func refreshControllerStartup(parent context.Context, refresh func(context.Context) error, active func() error, warning func(error)) error {
	ctx, cancel := context.WithTimeout(parent, controllerStartupTrustBudget)
	defer cancel()
	delay := 250 * time.Millisecond
	var last error
	for ctx.Err() == nil {
		call, stop := context.WithTimeout(ctx, controllerStartupTrustAttempt)
		err := refresh(call)
		stop()
		if err == nil {
			if err := parent.Err(); err != nil {
				return err
			}
			return active()
		}
		if !transientControllerTrust(err) {
			return err
		}
		last = err
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
		case <-timer.C:
		}
		if delay < 2*time.Second {
			delay *= 2
		}
	}
	if err := parent.Err(); err != nil {
		return errors.Join(last, err)
	}
	failure := errors.Join(last, ctx.Err())
	if err := active(); err != nil {
		return errors.Join(failure, err)
	}
	if warning != nil {
		warning(failure)
	}
	return nil
}

func transientControllerTrust(err error) bool {
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
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !transientControllerTrust(cause) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		if cause := wrapped.Unwrap(); cause != nil {
			return transientControllerTrust(cause)
		}
	}
	if status, ok := err.(*enrollmentHTTPError); ok {
		switch status.Status {
		case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusInternalServerError,
			http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			return true
		default:
			return false
		}
	}
	return transientLearnerTransport(err)
}
