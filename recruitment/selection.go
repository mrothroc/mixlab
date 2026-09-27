// Package recruitment owns deterministic cohort selection and controller-side
// orchestration. Discovery supplies addresses only; authenticated application
// ports supply capabilities. This package cannot issue trust or launch a child.
package recruitment

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/trust"
)

const SelectionPolicy = "node_id_ascending_v1"

type Requirements struct {
	Cluster         string         `json:"cluster"`
	Count           int            `json:"count"`
	BuildID         string         `json:"build_id"`
	MLXVersion      string         `json:"mlx_version"`
	DeviceKind      string         `json:"device_kind"`
	DType           string         `json:"dtype"`
	CustomOps       []string       `json:"custom_ops"`
	DatasetSelector string         `json:"dataset_selector"`
	DatasetID       string         `json:"dataset_id"`
	Limits          nodejob.Limits `json:"limits"`
	ProbeMaxAge     time.Duration  `json:"probe_max_age_ns"`
}

func canonicalHex(s string, n int) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == n && hex.EncodeToString(b) == s
}

func (r Requirements) Validate() error {
	if !canonicalHex(r.Cluster, 16) || r.Count < 2 || r.Count > 64 || !canonicalHex(r.BuildID, 32) || !canonicalHex(r.DatasetID, 32) || r.MLXVersion == "" || len(r.MLXVersion) > 128 || (r.DeviceKind != "metal" && r.DeviceKind != "cuda") || (r.DType != "fp32" && r.DType != "bf16") || r.DatasetSelector == "" || len(r.DatasetSelector) > 128 || r.ProbeMaxAge <= 0 || r.ProbeMaxAge > time.Hour {
		return fmt.Errorf("invalid cohort requirements")
	}
	if r.CustomOps == nil || len(r.CustomOps) > 256 || !slices.IsSorted(r.CustomOps) {
		return fmt.Errorf("explicit ordered custom-op set required")
	}
	for i, op := range r.CustomOps {
		if op == "" || len(op) > 128 || i > 0 && op == r.CustomOps[i-1] {
			return fmt.Errorf("invalid custom-op requirement")
		}
	}
	return r.Limits.Validate()
}

// Observation must come from a fresh, pinned-cluster authenticated response,
// never from TXT claims or JSON-supplied identity evidence. Lookup owns TLS and
// current-trust verification; recruitment still checks role/cluster/node/time.
type Observation struct {
	Peer         trust.AuthenticatedPrincipal
	Capabilities nodeagent.Capabilities
}
type Lookup func(context.Context, string) (Observation, error)

type Candidate struct {
	Endpoint     string                 `json:"endpoint"`
	Capabilities nodeagent.Capabilities `json:"capabilities"`
}
type Decision struct {
	Endpoint string `json:"endpoint"`
	Node     string `json:"node,omitempty"`
	Reason   string `json:"reason"`
}
type Selection struct {
	Policy       string       `json:"policy"`
	Requirements Requirements `json:"requirements"`
	Selected     []Candidate  `json:"selected"`
	Decisions    []Decision   `json:"decisions"`
}

// Select evaluates a bounded set of hints in endpoint order, then assigns ranks
// by authenticated node ID. No partial cohort is returned on failure. Decisions
// are stable reason codes, safe to log without leaking transport error details.
func Select(ctx context.Context, hints []discovery.Hint, r Requirements, lookup Lookup, clock func() time.Time) (Selection, error) {
	out := Selection{Policy: SelectionPolicy, Requirements: r, Selected: []Candidate{}, Decisions: []Decision{}}
	if err := r.Validate(); err != nil {
		return out, err
	}
	if lookup == nil || clock == nil || len(hints) > discovery.MaxHints {
		return out, fmt.Errorf("bounded discovery hints, authenticated lookup and clock required")
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	// Discard all claims before validation/query. Even a forged cluster/node
	// label cannot suppress or promote an otherwise valid authenticated endpoint.
	endpoints := map[string]bool{}
	for _, h := range hints {
		if h.Service != discovery.Node {
			continue
		}
		_, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: {h.Endpoint}}}).Browse(ctx, discovery.Node)
		if err == nil {
			endpoints[h.Endpoint] = true
		}
	}
	ordered := make([]string, 0, len(endpoints))
	for e := range endpoints {
		ordered = append(ordered, e)
	}
	sort.Strings(ordered)
	type observedCandidate struct {
		candidate Candidate
		peer      trust.AuthenticatedPrincipal
	}
	byNode := map[string][]observedCandidate{}
	excluded := map[string]string{}
	for _, endpoint := range ordered {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		query, stop := context.WithTimeout(ctx, 3*time.Second)
		o, err := lookup(query, endpoint)
		queryErr := query.Err()
		stop()
		d := Decision{Endpoint: endpoint}
		if err != nil || queryErr != nil {
			d.Reason = "unavailable_or_unauthenticated"
		} else {
			d.Reason = reject(o, r, clock())
			if d.Reason != "wrong_identity" {
				d.Node = o.Peer.Principal
				if d.Reason != "" {
					excluded[d.Node] = d.Reason
				}
			}
		}
		if d.Reason == "" {
			c := Candidate{endpoint, o.Capabilities}
			if previous, ok := byNode[d.Node]; ok {
				if stableCapabilityHash(previous[0].candidate.Capabilities) != stableCapabilityHash(c.Capabilities) {
					excluded[d.Node] = "conflicting_node_observations"
					d.Reason = "conflicting_node_observations"
				} else {
					d.Reason = "duplicate_node"
				}
			} else {
				d.Reason = "eligible"
			}
			byNode[d.Node] = append(byNode[d.Node], observedCandidate{c, o.Peer})
		}
		out.Decisions = append(out.Decisions, d)
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	var candidates []Candidate
	commitTime := clock()
	for node, observations := range byNode {
		if excluded[node] != "" {
			continue
		}
		// Endpoint order is deterministic, but only a still-live observation
		// can represent a multi-address node at commit time.
		for _, o := range observations {
			excluded[node] = reject(Observation{o.peer, o.candidate.Capabilities}, r, commitTime)
			if excluded[node] == "" {
				candidates = append(candidates, o.candidate)
				break
			}
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Capabilities.Node < candidates[j].Capabilities.Node })
	if len(candidates) >= r.Count {
		out.Selected = append(out.Selected, candidates[:r.Count]...)
	}
	for i := range out.Decisions {
		d := &out.Decisions[i]
		if reason := excluded[d.Node]; reason != "" {
			d.Reason = reason
		} else if d.Reason == "eligible" || d.Reason == "duplicate_node" {
			d.Reason = "not_selected"
			for _, c := range out.Selected {
				if c.Endpoint == d.Endpoint {
					d.Reason = "selected"
				}
			}
		}
	}
	if len(candidates) < r.Count {
		return out, fmt.Errorf("need %d compatible available nodes; found %d", r.Count, len(candidates))
	}
	return out, nil
}

func stableCapabilityHash(c nodeagent.Capabilities) string {
	c.ObservedAt = 0
	c.ProbeObservedAt = 0
	// Free memory fluctuates between sequential probes. Both observations must
	// independently satisfy the budget; prepare rechecks it after reservation.
	c.Probe.FreeMemoryBytes = nil
	b, _ := json.Marshal(c)
	return nodejob.Hash(b)
}

func reject(o Observation, r Requirements, now time.Time) string {
	p, c := o.Peer, o.Capabilities
	if p.Role != trust.Node || p.Cluster != r.Cluster || !canonicalHex(p.Principal, 16) || p.Principal != c.Node || now.Before(p.NotBefore) || !now.Before(p.ExpiresAt) {
		return "wrong_identity"
	}
	if c.Validate() != nil {
		return "invalid_capabilities"
	}
	if !c.Fresh(now, 30*time.Second, r.ProbeMaxAge) {
		return "stale_capabilities"
	}
	if !c.Recruitable {
		return "not_recruitable"
	}
	pb := c.Probe
	if pb.BuildID != r.BuildID || pb.MLXVersion != r.MLXVersion || pb.DeviceKind != r.DeviceKind || !slices.Contains(pb.DTypes, r.DType) || !slices.Equal(pb.CustomOps, r.CustomOps) || !slices.Contains(c.ManifestVersions, nodejob.Version) {
		return "incompatible_worker"
	}
	if !r.Limits.Within(c.Limits) || pb.MemoryBytes != nil && *pb.MemoryBytes < r.Limits.MemoryBytes || pb.FreeMemoryBytes != nil && *pb.FreeMemoryBytes < r.Limits.MemoryBytes {
		return "insufficient_resources"
	}
	if !slices.Contains(c.Datasets, nodeagent.Dataset{Selector: r.DatasetSelector, ID: r.DatasetID}) {
		return "dataset_mismatch"
	}
	return ""
}
