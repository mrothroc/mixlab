package clusterapp

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/learnerapi"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

func TestLearnerSemanticTransportRetryClassification(t *testing.T) {
	for _, err := range []error{syscall.ECONNREFUSED, syscall.ECONNRESET, syscall.ECONNABORTED, syscall.EPIPE, io.EOF, io.ErrUnexpectedEOF, context.DeadlineExceeded, &net.DNSError{IsTimeout: true}} {
		wrapped := &url.Error{Op: "Post", URL: "https://coordinator.invalid", Err: &net.OpError{Op: "read", Net: "tcp", Err: err}}
		if !transientLearnerTransport(wrapped) {
			t.Errorf("did not retry transport failure: %v", err)
		}
	}
	for _, err := range []error{
		nil, context.Canceled, syscall.EACCES, syscall.EHOSTUNREACH, &net.DNSError{IsNotFound: true},
		tls.AlertError(42), &tls.CertificateVerificationError{Err: context.DeadlineExceeded},
		x509.UnknownAuthorityError{}, x509.CertificateInvalidError{}, x509.HostnameError{},
		fmt.Errorf("%w: %w", managedtls.ErrPeerRejected, io.EOF),
		&net.OpError{Op: "remote error", Err: io.EOF}, errors.New("HTTP 403"), errors.New("malformed response"),
	} {
		if transientLearnerTransport(err) {
			t.Errorf("retried non-transport failure: %v", err)
		}
	}
}

func TestLearnerSemanticRetryBudgetAndCancellation(t *testing.T) {
	t.Run("retry", func(t *testing.T) {
		calls := 0
		_, err := retryLearnerSemantic(context.Background(), func(ctx context.Context) (learnerapi.Response, bool, error) {
			calls++
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) > learnerAttemptBudget {
				t.Fatal("unbounded attempt")
			}
			if calls == 1 {
				return learnerapi.Response{}, true, io.EOF
			}
			return learnerapi.Response{}, false, nil
		})
		if err != nil || calls != 2 {
			t.Fatal(calls, err)
		}
	})
	t.Run("nonretryable", func(t *testing.T) {
		calls := 0
		want := errors.New("policy unavailable")
		_, err := retryLearnerSemantic(context.Background(), func(context.Context) (learnerapi.Response, bool, error) {
			calls++
			return learnerapi.Response{}, false, want
		})
		if !errors.Is(err, want) || calls != 1 {
			t.Fatal(calls, err)
		}
	})
	t.Run("caller-budget", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		defer cancel()
		calls := 0
		start := time.Now()
		_, err := retryLearnerSemantic(ctx, func(context.Context) (learnerapi.Response, bool, error) {
			calls++
			return learnerapi.Response{}, true, syscall.ECONNREFUSED
		})
		if !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, syscall.ECONNREFUSED) || calls != 1 || time.Since(start) > time.Second {
			t.Fatal(calls, err)
		}
	})
	t.Run("already-canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := retryLearnerSemantic(ctx, func(context.Context) (learnerapi.Response, bool, error) {
			t.Fatal("attempt after cancellation")
			return learnerapi.Response{}, false, nil
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	})
}
