package clusterapp

import (
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
)

func TestNodeCheckpointOperationDeadline(t *testing.T) {
	job := "/v1/agent/jobs/" + strings.Repeat("a", 32)
	for _, tc := range []struct {
		method, path string
		want         time.Duration
	}{
		{http.MethodPut, job + "/input", 50 * time.Second},
		{http.MethodPut, job + "/prepare", 50 * time.Second},
		{http.MethodPost, job + "/start", 50 * time.Second},
		{http.MethodPut, job + "/transport", 10 * time.Second},
		{http.MethodGet, job + "/input", 10 * time.Second},
		{http.MethodPut, "/v1/agent/jobs/bad/input", 10 * time.Second},
		{http.MethodPost, "/v1/agent/leases", 10 * time.Second},
	} {
		if got := nodeOperationTimeout(tc.method, tc.path); got != tc.want {
			t.Fatal(tc, got)
		}
		// Construction only: no network or credentials. Keep route budgets within
		// the transport API's independent ceiling rather than bypassing it.
		p := new(managedtls.Policy)
		client, err := p.HTTPClient(tc.want + 5*time.Second)
		if err != nil {
			t.Fatal("route exceeds transport deadline contract", tc, err)
		}
		client.CloseIdleConnections()
	}
}
