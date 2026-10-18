package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

type workloadRevokingBody struct {
	io.Reader
	once   sync.Once
	revoke func()
}

func (b *workloadRevokingBody) Read(p []byte) (int, error) {
	b.once.Do(b.revoke)
	return b.Reader.Read(p)
}

// Exercise the handler with genuine principal evidence, injecting revocation
// during decoding after middleware authentication. Socket/TLS behavior is
// covered separately by TestWorkloadAuthorizedIssuanceOverManagedTLS.
func TestWorkloadIssuanceRechecksTrustAfterBody(t *testing.T) {
	for _, route := range []string{workloadIssueRoute, workloadIssueAuthorizedRoute} {
		for _, role := range []string{"node", "controller"} {
			t.Run(route+"/"+role, func(t *testing.T) {
				f := newRecruitmentAuthorizationFixture(t)
				ctx := context.Background()
				auth := f.authorization(t)
				grant, err := f.signer.grantWithAuthorization(ctx, f.request, auth)
				check(t, err)
				body := recruitmentAuthorizationJSON(t, grant)
				if route == workloadIssueAuthorizedRoute {
					envelope, err := f.signer.authorizeGrant(ctx, grant, auth)
					check(t, err)
					body = recruitmentAuthorizationJSON(t, envelope)
				}
				identity, key, err := f.a.Principal.Active(f.now)
				check(t, err)
				policy, err := managedtls.New(managedtls.Identity{Chain: identity.Chain, Key: key}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
					view, err := f.a.Current(ctx, now)
					if err != nil {
						return trust.AuthenticatedPrincipal{}, err
					}
					return trust.AuthenticatePrincipal(f.a.Anchor, view, chain, now)
				}, func() time.Time { return f.now })
				check(t, err)
				cs, _, err := f.controller.Active(f.now)
				check(t, err)
				state := &tls.ConnectionState{HandshakeComplete: true, Version: tls.VersionTLS13, NegotiatedProtocol: managedtls.Protocol}
				for _, der := range cs.Chain {
					cert, err := x509.ParseCertificate(der)
					check(t, err)
					state.PeerCertificates = append(state.PeerCertificates, cert)
				}
				revoked := false
				input := &workloadRevokingBody{Reader: bytes.NewReader(body), revoke: func() {
					id := f.request.Request.Scope.Node
					if role == "controller" {
						id = cs.Principal
					}
					_, err := f.a.Snapshots.Revoke(ctx, "principal", id, "prospective", "body decode regression", f.now)
					check(t, err)
					revoked = true
				}}
				r := httptest.NewRequest(http.MethodPost, route, input)
				r.Header.Set("Content-Type", "application/json")
				r.TLS = state
				w := httptest.NewRecorder()
				policy.AuthenticateHTTP(authorityManagedHandler(f.a, func() time.Time { return f.now })).ServeHTTP(w, r)
				if !revoked || w.Code != http.StatusForbidden {
					t.Fatalf("issued against pre-body trust: revoked=%v status=%d body=%s", revoked, w.Code, w.Body.String())
				}
			})
		}
	}
}
