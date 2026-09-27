package clusterapp

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

type consumeRequest struct {
	Request enrollment.SignedRequest `json:"request"`
	Secret  []byte                   `json:"secret"`
}

// ServeInvitation is a bounded, one-invitation temporary source. It closes only
// after the consuming connection ends (so successful replies can be flushed).
// The input invitation never goes on a public metadata route.
func ServeInvitation(ctx context.Context, listener net.Listener, a *Authority, i enrollment.Invitation, clock func() time.Time) error {
	if listener == nil || a == nil || clock == nil {
		return fmt.Errorf("provisioning source dependencies required")
	}
	if _, err := i.ValidateTarget(i.Endpoint, i.Audience, clock()); err != nil {
		return err
	}
	if _, err := a.Current(ctx, clock()); err != nil {
		return err
	}
	state, key, err := a.Principal.Active(clock())
	if err != nil {
		return err
	}
	life, cancel := context.WithCancel(ctx)
	defer cancel()
	var consumed atomic.Pointer[[32]byte]
	return serveBootstrap(life, listener, enrollmenttls.Identity{Chain: state.Chain, Key: key}, time.Unix(i.ExpiresAt, 0), clock, func(c *enrollmenttls.Channel) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.TLS == nil || r.TLS.NegotiatedProtocol != enrollmenttls.Protocol {
				http.Error(w, "bootstrap channel required", http.StatusUnauthorized)
				return
			}
			if _, err := c.Network(); err != nil {
				http.Error(w, "connection expired", http.StatusUnauthorized)
				return
			}
			if r.URL.RawQuery != "" {
				http.Error(w, "unexpected query", 400)
				return
			}
			v, err := a.Current(r.Context(), clock())
			if err != nil {
				http.Error(w, "authority unavailable", http.StatusServiceUnavailable)
				return
			}
			if r.Method == "GET" && r.URL.Path == "/v1/trust/snapshots/latest" {
				b, err := v.Bytes()
				if err != nil {
					http.Error(w, "trust unavailable", http.StatusServiceUnavailable)
					return
				}
				writeJSON(w, json.RawMessage(b))
				return
			}
			if r.Method != "POST" || r.URL.Path != "/v1/trust/enrollments/"+i.ID+"/consume" {
				http.NotFound(w, r)
				return
			}
			var q consumeRequest
			defer func() { clear(q.Secret) }()
			if err := readJSON(w, r, &q, 16<<10); err != nil {
				http.Error(w, "invalid enrollment request", 400)
				return
			}
			if q.Request.Request.Invitation != i.ID {
				http.Error(w, "invitation mismatch", 400)
				return
			}
			out, err := a.Enrollment.Consume(r.Context(), q.Request, q.Secret, v, clock())
			if err != nil {
				http.Error(w, "enrollment rejected; do not blindly retry a possibly consumed file", 400)
				return
			}
			id := c.EvidenceID()
			consumed.Store(&id)
			w.Header().Set("Connection", "close")
			writeJSON(w, out)
		})
	}, func(_ context.Context, c *enrollmenttls.Channel) error {
		if id := consumed.Load(); id != nil && *id == c.EvidenceID() {
			cancel()
		}
		return nil
	})
}
