// Package snapshottls provides a route-confined public signed-snapshot client.
// It never sends client credentials or performs a renewal/application request.
package snapshottls

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/mrothroc/mixlab/transport/internal/tlsidentity"
)

type VerifySource func([][]byte, time.Time) error

func HTTPClient(endpoint string, verify VerifySource, clock func() time.Time, timeout time.Duration) (*http.Client, error) {
	r, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil || r.URL.Scheme != "https" || r.URL.Hostname() == "" || r.URL.User != nil || r.URL.RawQuery != "" || r.URL.Fragment != "" || (r.URL.Path != "" && r.URL.Path != "/") || verify == nil || clock == nil || timeout <= 0 || timeout > time.Minute {
		return nil, fmt.Errorf("bounded pinned snapshot client and HTTPS origin required")
	}
	c := &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13, NextProtos: []string{"http/1.1"}, SessionTicketsDisabled: true, Time: clock,
		InsecureSkipVerify: true, // mandatory pinned source verification below; never Web PKI fallback
		VerifyConnection: func(s tls.ConnectionState) error {
			if s.Version != tls.VersionTLS13 || s.DidResume || s.NegotiatedProtocol != "http/1.1" {
				return fmt.Errorf("invalid snapshot TLS profile")
			}
			chain, err := tlsidentity.PeerChain(s)
			if err != nil {
				return err
			}
			return verify(chain, clock())
		},
	}
	t := &transport{origin: r.URL.Host, inner: &http.Transport{TLSClientConfig: c, TLSHandshakeTimeout: timeout, DialContext: (&net.Dialer{Timeout: timeout}).DialContext, DisableKeepAlives: true, ResponseHeaderTimeout: timeout, MaxResponseHeaderBytes: 8 << 10}}
	return &http.Client{Transport: t, Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}, nil
}

type transport struct {
	origin string
	inner  *http.Transport
}

func (t *transport) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL == nil || r.URL.Scheme != "https" || r.URL.Host != t.origin || (r.Host != "" && r.Host != t.origin) || r.URL.User != nil || r.URL.RawQuery != "" || r.URL.Fragment != "" {
		return nil, fmt.Errorf("snapshot transport cannot perform application operations or change origin")
	}
	allowed := (r.Method == http.MethodGet && r.URL.Path == "/v1/trust/snapshots/latest") || (r.Method == http.MethodPut && r.URL.Path == "/v1/trust/snapshot")
	if !allowed {
		return nil, fmt.Errorf("snapshot transport cannot perform application operations")
	}
	return t.inner.RoundTrip(r)
}
func (t *transport) CloseIdleConnections() { t.inner.CloseIdleConnections() }
