package managedtls

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

type principalContextKey struct{}

// Principal returns identity evidence installed by AuthenticateHTTP, not an
// application grant. Handlers must still authorize their context-owned action.
func Principal(ctx context.Context) (trust.AuthenticatedPrincipal, bool) {
	id, ok := ctx.Value(principalContextKey{}).(trust.AuthenticatedPrincipal)
	return id, ok
}

type clientTransport struct{ inner *http.Transport }

func (t *clientTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL == nil || r.URL.Scheme != "https" || r.URL.Host == "" || r.URL.User != nil || r.URL.Fragment != "" {
		return nil, fmt.Errorf("managed requests require an explicit HTTPS endpoint")
	}
	return t.inner.RoundTrip(r)
}
func (t *clientTransport) CloseIdleConnections() { t.inner.CloseIdleConnections() }

// HTTPClient revalidates the peer on every request by using a new full TLS
// connection. It never follows redirects or environment proxies. Higher-volume
// adapters may reuse connections only with an equivalent pre-write trust check.
func (p *Policy) HTTPClient(timeout time.Duration) (*http.Client, error) {
	if p == nil || timeout <= 0 || timeout > time.Minute {
		return nil, fmt.Errorf("managed HTTP timeout must be in (0,1m]")
	}
	tr := &clientTransport{inner: &http.Transport{
		TLSClientConfig: p.ClientConfig(), TLSHandshakeTimeout: timeout,
		DialContext:       (&net.Dialer{Timeout: timeout}).DialContext,
		DisableKeepAlives: true, ResponseHeaderTimeout: timeout, MaxResponseHeaderBytes: 16 << 10,
	}}
	return &http.Client{Transport: tr, Timeout: timeout,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}, nil
}

// AuthenticateHTTP fails closed before invoking application code, including
// on keepalive requests after snapshot expiry or revocation. It exposes no
// certificate/verification failure details to unauthenticated peers.
func (p *Policy) AuthenticateHTTP(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil || next == nil {
			http.Error(w, "managed TLS required", http.StatusUnauthorized)
			return
		}
		id, err := p.Authenticate(*r.TLS)
		if err != nil {
			http.Error(w, "managed TLS authentication failed", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), principalContextKey{}, id)))
	})
}
