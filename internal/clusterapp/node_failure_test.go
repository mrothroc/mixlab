package clusterapp

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"syscall"
	"testing"

	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/transport/managedtls"
)

func TestNodeRequestFailureRetainsIdentityAndClassification(t *testing.T) {
	id := strings.Repeat("a", 32)
	c, err := NewNodeClientWithPolicy("127.0.0.1:7445", id, func() (*managedtls.Policy, error) { return nil, syscall.EHOSTUNREACH })
	check(t, err)
	err = c.request(context.Background(), http.MethodGet, "/v1/agent/leases", nil, nil)
	var scoped interface{ NodeID() string }
	if !errors.As(err, &scoped) || scoped.NodeID() != id {
		t.Fatalf("node identity lost: %v", err)
	}
	if got := clusterdiagnostic.Classify(err); got.Reason != "route_or_permission_denied" {
		t.Fatalf("classification lost: %+v", got)
	}
}
