package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRecoveryAuthorityHTTPStatus(t *testing.T) {
	for _, status := range []int{400, 401, 403, 409, 429, 500, 502, 503, 504} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(status)
				_, _ = w.Write([]byte("private authority diagnostic"))
			}))
			defer server.Close()
			var out any
			err := enrollmentJSON(context.Background(), server.Client(), "GET", server.URL+"/v1/trust/snapshots/latest", nil, &out)
			var h *enrollmentHTTPError
			if !errors.As(err, &h) || h.Status != status || strings.Contains(err.Error(), "private authority diagnostic") {
				t.Fatalf("lost status or exposed body: %v", err)
			}
			want := status == 429 || status == 502 || status == 503 || status == 504
			if recoveryTransient(fmt.Errorf("refresh recovery authority trust: %w", err)) != want {
				t.Fatalf("HTTP %d retry classification differs", status)
			}
		})
	}
}
