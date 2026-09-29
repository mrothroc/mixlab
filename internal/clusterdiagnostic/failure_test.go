package clusterdiagnostic

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"strings"
	"syscall"
	"testing"
)

func TestFailureSafeClassification(t *testing.T) {
	for _, tt := range []struct {
		err    error
		reason string
	}{
		{syscall.EHOSTUNREACH, "route_or_permission_denied"}, {syscall.ECONNREFUSED, "connection_refused"},
		{context.DeadlineExceeded, "timeout"}, {context.Canceled, "canceled"}, {tls.AlertError(42), "tls_rejected"},
		{&net.DNSError{Name: "secret", Err: "private"}, "dns_failed"}, {errors.New("/secret/private-key"), "unavailable_or_unauthenticated"},
	} {
		f := Classify(fmt.Errorf("secret: %w", tt.err))
		if f.Reason != tt.reason || strings.Contains(f.Hint, "secret") {
			t.Fatal(f)
		}
	}
	if !strings.Contains(Classify(syscall.EHOSTUNREACH).Hint, "does not prove") {
		t.Fatal("false definitive LNP diagnosis")
	}
}
