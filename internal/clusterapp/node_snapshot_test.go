package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func assertSnapshotRouteIsolation(t *testing.T, ctx context.Context, endpoint string, clock func() time.Time, snapshot trust.SignedSnapshot) {
	t.Helper()
	// Deliberately unauthenticated test client, bypassing the route-confined
	// production client to verify the server's boundary independently.
	tr := &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true, MinVersion: tls.VersionTLS13, NextProtos: []string{"http/1.1"}, Time: clock}, DisableKeepAlives: true}
	defer tr.CloseIdleConnections()
	client := &http.Client{Transport: tr, Timeout: 3 * time.Second}
	bad := snapshot
	bad.Signature = bytes.Clone(snapshot.Signature)
	bad.Signature[0] ^= 1
	b, err := json.Marshal(bad)
	check(t, err)
	valid, err := json.Marshal(snapshot)
	check(t, err)
	for _, tc := range []struct {
		method, path string
		body         []byte
		status       int
	}{
		{http.MethodGet, nodeCapabilityRoute, nil, http.StatusUnauthorized},
		{http.MethodPost, "/v1/agent/leases", []byte(`{}`), http.StatusUnauthorized},
		{http.MethodPut, nodeSnapshotRoute, b, http.StatusConflict},
		{http.MethodPut, nodeSnapshotRoute + "?extra=1", valid, http.StatusNotFound},
		{http.MethodGet, nodeSnapshotRoute, nil, http.StatusNotFound},
		{http.MethodPut, nodeSnapshotRoute, []byte(`{"unexpected":true}`), http.StatusBadRequest},
	} {
		r, err := http.NewRequestWithContext(ctx, tc.method, "https://"+endpoint+tc.path, bytes.NewReader(tc.body))
		check(t, err)
		r.Header.Set("Content-Type", "application/json")
		response, err := client.Do(r)
		check(t, err)
		check(t, response.Body.Close())
		if response.StatusCode != tc.status {
			t.Fatalf("%s %s: %d want %d", tc.method, tc.path, response.StatusCode, tc.status)
		}
	}
}
