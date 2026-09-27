package clusterapp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/strictjson"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

const nodeCapabilityRoute = "/v1/agent/capabilities"
const maxCapabilityBytes = 256 << 10

// NodeCapabilitiesHandler serves only the read-only authenticated capability
// contract. The operational agent composition owns its listener and lifecycle.
func NodeCapabilitiesHandler(store *nodeagent.Store, policy *managedtls.Policy, clock func() time.Time) (http.Handler, error) {
	if store == nil || policy == nil || clock == nil {
		return nil, fmt.Errorf("node store, managed TLS and clock required")
	}
	return policy.AuthenticateHTTP(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != nodeCapabilityRoute || r.URL.RawQuery != "" || r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			http.NotFound(w, r)
			return
		}
		peer, ok := managedtls.Principal(r.Context())
		if !ok || peer.Role != trust.Controller {
			http.Error(w, "controller required", http.StatusForbidden)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		c, err := store.Capabilities(ctx, peer, clock())
		if err != nil || c.Validate() != nil {
			http.Error(w, "node capabilities unavailable", http.StatusServiceUnavailable)
			return
		}
		writeJSON(w, c)
	})), nil
}

// QueryNodeCapabilities binds the decoded response to this request's actual
// TLS peer, reauthenticated against current trust after body consumption too.
// The policy must require a Node in the controller's pinned cluster.
func QueryNodeCapabilities(ctx context.Context, endpoint string, policy *managedtls.Policy) (recruitment.Observation, error) {
	var out recruitment.Observation
	if policy == nil {
		return out, fmt.Errorf("managed node TLS policy required")
	}
	if _, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: {endpoint}}}).Browse(ctx, discovery.Node); err != nil {
		return out, err
	}
	c, err := policy.HTTPClient(3 * time.Second)
	if err != nil {
		return out, err
	}
	defer c.CloseIdleConnections()
	target := (&url.URL{Scheme: "https", Host: endpoint, Path: nodeCapabilityRoute}).String()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return out, err
	}
	response, err := c.Do(request)
	if err != nil {
		return out, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK || response.TLS == nil {
		return out, fmt.Errorf("node capability query rejected (HTTP %d)", response.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(response.Body, maxCapabilityBytes+1))
	if err != nil {
		return out, err
	}
	if len(b) > maxCapabilityBytes {
		return out, fmt.Errorf("node capabilities exceed response limit")
	}
	if err := strictjson.Validate(b, 16); err != nil {
		return out, err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&out.Capabilities); err != nil {
		return out, err
	}
	if err := out.Capabilities.Validate(); err != nil {
		return out, err
	}
	out.Peer, err = policy.Authenticate(*response.TLS)
	if err != nil {
		return recruitment.Observation{}, err
	}
	if out.Peer.Role != trust.Node || out.Peer.Principal != out.Capabilities.Node {
		return recruitment.Observation{}, fmt.Errorf("capabilities do not belong to authenticated node")
	}
	return out, nil
}

// NodeLookup creates a fresh policy per query from the controller's protected
// current credentials. Discovery cannot replace the pin or choose peer roles.
func NodeLookup(p *principal.Store, clock func() time.Time) (recruitment.Lookup, error) {
	if p == nil || clock == nil {
		return nil, fmt.Errorf("controller principal and clock required")
	}
	return func(ctx context.Context, endpoint string) (recruitment.Observation, error) {
		s, _, err := p.Active(clock())
		if err != nil {
			return recruitment.Observation{}, err
		}
		if s.Role != trust.Controller {
			return recruitment.Observation{}, fmt.Errorf("controller principal required")
		}
		policy, err := managedPrincipalPolicy(p, trust.Node, clock)
		if err != nil {
			return recruitment.Observation{}, err
		}
		return QueryNodeCapabilities(ctx, endpoint, policy)
	}, nil
}
