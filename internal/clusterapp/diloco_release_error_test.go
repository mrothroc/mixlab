package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/mrothroc/mixlab/coordinator/nodeleases"
)

func TestClassifyLeaseReleaseError(t *testing.T) {
	for _, status := range []int{http.StatusConflict, http.StatusForbidden, http.StatusUnauthorized, http.StatusServiceUnavailable} {
		remote := &NodeHTTPError{Status: status}
		for _, original := range []error{remote, fmt.Errorf("release: %w", remote)} {
			got := classifyLeaseReleaseError(original)
			if errors.Is(got, nodeleases.ErrReleaseConflict) != (status == http.StatusConflict) || !errors.Is(got, original) {
				t.Fatalf("status %d incorrectly classified: %v", status, got)
			}
			var preserved *NodeHTTPError
			if !errors.As(got, &preserved) || preserved != remote {
				t.Fatal("lost HTTP diagnostic")
			}
		}
	}
	for _, original := range []error{nil, context.Canceled, context.DeadlineExceeded, errors.New("node operation rejected (HTTP 409)")} {
		if got := classifyLeaseReleaseError(original); got != original {
			t.Fatalf("non-HTTP error changed: %v", got)
		}
	}
}
