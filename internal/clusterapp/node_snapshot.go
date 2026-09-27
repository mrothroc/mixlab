package clusterapp

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/transport/snapshottls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

const nodeSnapshotRoute = "/v1/trust/snapshot"

func snapshotDigest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

// NodeSnapshots accepts only public, signed trust-state advancement. Possession
// of a transport connection confers no authority; the pinned root and snapshot
// signer authorize installation. Normal node routes retain current mutual TLS.
type NodeSnapshots struct {
	Principal *principal.Store
	Clock     func() time.Time
}
type SnapshotReceipt struct {
	Node       string `json:"node"`
	Generation uint64 `json:"generation"`
	Digest     string `json:"digest"`
}

func (s *NodeSnapshots) Policy() (*managedtls.Policy, error) {
	if s == nil || s.Principal == nil || s.Clock == nil {
		return nil, fmt.Errorf("node snapshot owner and clock required")
	}
	r, key, err := s.Principal.SnapshotReceiver(s.Clock())
	if err != nil {
		return nil, err
	}
	a, err := trust.PinRoot(r.Root, r.Fingerprint, s.Clock())
	if err != nil {
		return nil, err
	}
	return managedtls.New(managedtls.Identity{Chain: r.Chain, Key: key}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		current, _, err := s.Principal.Active(now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		v, err := trust.VerifySnapshot(a, current.Snapshot, now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		peer, err := trust.AuthenticatePrincipal(a, v, chain, now)
		if err == nil && peer.Role != trust.Controller {
			err = fmt.Errorf("controller required")
		}
		return peer, err
	}, s.Clock)
}

func (s *NodeSnapshots) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut || r.URL.Path != nodeSnapshotRoute || r.URL.RawPath != "" || r.URL.RawQuery != "" || r.TLS == nil {
		http.NotFound(w, r)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	var next trust.SignedSnapshot
	if err := readJSON(w, r.WithContext(ctx), &next, trust.MaxTrustBytes); err != nil {
		http.Error(w, "invalid bounded snapshot", http.StatusBadRequest)
		return
	}
	if _, _, err := s.Principal.SnapshotReceiver(s.Clock()); err != nil {
		http.Error(w, "node identity unavailable", http.StatusServiceUnavailable)
		return
	}
	if err := refreshPrincipal(ctx, s.Principal, next, s.Clock()); err != nil {
		http.Error(w, "snapshot rejected", http.StatusConflict)
		return
	}
	current, err := s.Principal.View(s.Clock())
	if err != nil {
		http.Error(w, "snapshot unavailable", http.StatusServiceUnavailable)
		return
	}
	b, err := json.Marshal(current.Snapshot)
	if err != nil {
		http.Error(w, "snapshot unavailable", http.StatusServiceUnavailable)
		return
	}
	writeJSON(w, SnapshotReceipt{Node: current.Principal, Generation: current.Snapshot.Payload.Generation, Digest: snapshotDigest(b)})
}

// PushNodeSnapshot sends no secret or client credential, only a prevalidated
// public signed snapshot. Discovery selects routes, never the root or role.
func PushNodeSnapshot(ctx context.Context, anchor trust.Anchor, next trust.SignedSnapshot, endpoint string, clock func() time.Time) (SnapshotReceipt, error) {
	var out SnapshotReceipt
	if clock == nil {
		return out, fmt.Errorf("clock required")
	}
	if _, err := trust.VerifySnapshot(anchor, next, clock()); err != nil {
		return out, err
	}
	if _, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: {endpoint}}}).Browse(ctx, discovery.Node); err != nil {
		return out, err
	}
	origin := (&url.URL{Scheme: "https", Host: endpoint}).String()
	var peer string
	c, err := snapshottls.HTTPClient(origin, func(chain [][]byte, now time.Time) error {
		var err error
		peer, err = trust.VerifySnapshotReceiver(anchor, chain, now)
		return err
	}, clock, 15*time.Second)
	if err != nil {
		return out, err
	}
	defer c.CloseIdleConnections()
	if err := enrollmentJSON(ctx, c, http.MethodPut, origin+nodeSnapshotRoute, next, &out); err != nil {
		return out, err
	}
	b, err := json.Marshal(next)
	if err != nil {
		return out, err
	}
	if peer == "" || out.Node != peer || out.Generation != next.Payload.Generation || out.Digest != snapshotDigest(b) {
		return out, fmt.Errorf("snapshot receipt differs from transmitted state or TLS peer")
	}
	return out, nil
}
