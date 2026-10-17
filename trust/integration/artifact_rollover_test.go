package integration

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

func TestArtifactGatewayGrantRollover(t *testing.T) {
	for _, mode := range []string{"catch-up", "nil-refresh", "unchanged-refresh", "failed-refresh", "forged-proof", "forged-snapshot", "wrong-assignment", "revoked-after-refresh", "revoked-caller-after-refresh"} {
		t.Run(mode, func(t *testing.T) {
			body := []byte("artifact behind independently refreshed trust")
			a := newArtifactFixture(t, artifactRef(body))
			check(t, a.g.Store.PutVerified(context.Background(), artifactRef(body), bytes.NewReader(body)))
			local := &rolloverTrust{view: a.f.view}
			peer := advanceRolloverSigner(t, a.f)
			q := a.request(t, 0, "read", "rollover-read")
			if q.Grant.Proof.Evidence.Snapshot.Payload.Generation != peer.Generation() {
				t.Fatal("fixture did not issue ahead-of-gateway proof")
			}
			latest := rolloverSnapshot(t, a.f, peer.Generation()+1, nil)
			if mode == "revoked-after-refresh" {
				latest = rolloverSnapshot(t, a.f, peer.Generation()+1, []trust.Revocation{{Kind: "principal", ID: a.f.serverID, Mode: "compromise", Reason: "retired coordinator", FirstGeneration: peer.Generation() + 1}})
			}
			if mode == "revoked-caller-after-refresh" {
				latest = rolloverSnapshot(t, a.f, peer.Generation()+1, []trust.Revocation{{Kind: "principal", ID: a.clientIDs[0], Mode: "compromise", Reason: "retired node", FirstGeneration: peer.Generation() + 1}})
			}
			switch mode {
			case "wrong-assignment":
				body := q.Grant.Body
				body.AssignmentID = "different-assignment"
				q = a.sign(t, body)
			case "forged-proof":
				q.Grant.Proof.Signature[0] ^= 1
			case "forged-snapshot":
				q.Grant.Proof.Evidence.Snapshot.Signature[0] ^= 1
			}
			a.g.Trust = local
			var err error
			a.g.Policy, err = managedtls.New(a.f.server, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
				v, err := local.Current(context.Background(), now)
				if err != nil {
					return trust.AuthenticatedPrincipal{}, err
				}
				return trust.AuthenticatePrincipal(a.f.a, v, chain, now)
			}, a.f.clock)
			check(t, err)
			var refreshes atomic.Int32
			if mode != "nil-refresh" {
				a.g.RefreshTrust = func(ctx context.Context) error {
					refreshes.Add(1)
					if _, ok := ctx.Deadline(); !ok {
						t.Error("gateway refresh escaped transfer deadline")
					}
					switch mode {
					case "unchanged-refresh":
						return nil
					case "failed-refresh":
						return errors.New("pinned authority unavailable")
					default:
						local.install(latest)
						return nil
					}
				}
			}
			handler, err := a.g.Handler()
			check(t, err)
			server := serveCapabilities(t, a.g.Policy, handler)
			policy, err := managedtls.New(a.clients[0], a.f.verify(trust.Coordinator, a.f.serverID), a.f.clock)
			check(t, err)
			client, err := policy.HTTPClient(5 * time.Second)
			check(t, err)
			defer client.CloseIdleConnections()
			send := func() (int, []byte) {
				t.Helper()
				var encoded bytes.Buffer
				check(t, artifactapi.WriteRequest(&encoded, q))
				response, err := client.Post(server.URL+artifactapi.ReadRoute, "application/octet-stream", &encoded)
				check(t, err)
				defer func() { _ = response.Body.Close() }()
				out, err := io.ReadAll(response.Body)
				check(t, err)
				return response.StatusCode, out
			}
			wantStatus, wantRefreshes := http.StatusOK, int32(1)
			switch mode {
			case "nil-refresh":
				wantStatus, wantRefreshes = http.StatusServiceUnavailable, 0
			case "unchanged-refresh", "failed-refresh":
				wantStatus = http.StatusServiceUnavailable
			case "forged-proof", "forged-snapshot", "wrong-assignment":
				wantStatus, wantRefreshes = http.StatusForbidden, 0
			case "revoked-after-refresh", "revoked-caller-after-refresh":
				wantStatus = http.StatusForbidden
			}
			status, out := send()
			if status != wantStatus || refreshes.Load() != wantRefreshes {
				t.Fatalf("rollover classification: HTTP %d want=%d refreshes=%d want=%d body=%s", status, wantStatus, refreshes.Load(), wantRefreshes, out)
			}
			if status == http.StatusOK {
				if !bytes.Equal(out, body) {
					t.Fatal("catch-up changed artifact bytes")
				}
			} else if bytes.Contains(out, body) {
				t.Fatal("rejected grant exposed artifact bytes")
			}
			if status == http.StatusServiceUnavailable {
				v, err := local.Current(context.Background(), a.f.clock())
				check(t, err)
				if v.Generation() >= peer.Generation() {
					t.Fatal("gateway installed peer evidence on unavailable refresh")
				}
				// The exact grant/use is still usable after independent maintenance
				// catches up; the 503 must not permanently reject its use ID.
				local.install(latest)
				if status, out = send(); status != http.StatusOK || !bytes.Equal(out, body) {
					t.Fatalf("exact retry failed after maintenance caught up: HTTP %d body=%s", status, out)
				}
				if refreshes.Load() != wantRefreshes {
					t.Fatal("current trust triggered another refresh")
				}
			}
		})
	}
}
