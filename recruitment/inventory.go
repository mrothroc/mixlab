package recruitment

import (
	"context"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/trust"
)

// NodeStatus is an authenticated observation, not a reservation or a cohort
// approval. Multiple addresses for one node remain visible to the operator.
// Errors use reason codes rather than remote bodies or local credential paths.
type NodeStatus struct {
	Endpoint     string                  `json:"endpoint"`
	Reason       string                  `json:"reason"`
	Hint         string                  `json:"hint,omitempty"`
	Capabilities *nodeagent.Capabilities `json:"capabilities,omitempty"`
}

func Inventory(ctx context.Context, hints []discovery.Hint, cluster string, lookup Lookup, clock func() time.Time) ([]NodeStatus, error) {
	if !canonicalHex(cluster, 16) || lookup == nil || clock == nil || len(hints) > discovery.MaxHints {
		return nil, fmt.Errorf("pinned cluster, bounded hints, authenticated lookup and clock required")
	}
	addresses := make([]string, 0, len(hints))
	for _, h := range hints {
		if h.Service != discovery.Node {
			return nil, fmt.Errorf("node inventory received another discovery service")
		}
		addresses = append(addresses, h.Endpoint)
	}
	// Revalidate/deduplicate addresses and intentionally discard all TXT claims.
	ordered, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: addresses}}).Browse(ctx, discovery.Node)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out := make([]NodeStatus, 0, len(ordered))
	for _, h := range ordered {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		query, stop := context.WithTimeout(ctx, 3*time.Second)
		o, err := lookup(query, h.Endpoint)
		queryErr := query.Err()
		stop()
		s := NodeStatus{Endpoint: h.Endpoint, Reason: "unavailable_or_unauthenticated"}
		if err != nil || queryErr != nil {
			if queryErr != nil {
				err = queryErr
			}
			failure := clusterdiagnostic.Classify(err)
			s.Reason, s.Hint = failure.Reason, failure.Hint
		}
		if err == nil && queryErr == nil {
			p, c, now := o.Peer, o.Capabilities, clock()
			switch {
			case p.Cluster != cluster || p.Role != trust.Node || !canonicalHex(p.Principal, 16) || p.Principal != c.Node || now.Before(p.NotBefore) || !now.Before(p.ExpiresAt):
				s.Reason = "wrong_identity"
			case c.Validate() != nil:
				s.Reason = "invalid_capabilities"
			default:
				s.Capabilities = &c
				switch {
				case !c.Fresh(now, 30*time.Second, 2*time.Minute):
					s.Reason = "stale_capabilities"
				case !c.Recruitable:
					s.Reason = "not_recruitable"
					if !c.Availability.Available {
						s.Reason = "busy"
					}
				default:
					s.Reason = "available"
				}
			}
		}
		out = append(out, s)
	}
	return out, ctx.Err()
}
