package recruitment

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/trust"
)

func TestInventoryAuthenticatedBoundedDeterministic(t *testing.T) {
	r, o := fixture()
	hints := []discovery.Hint{{Service: discovery.Node, Endpoint: "z.local:1", Claims: discovery.Claims{Node: "forged"}}, {Service: discovery.Node, Endpoint: "a.local:1"}, {Service: discovery.Node, Endpoint: "z.local:1"}}
	calls := 0
	lookup := func(ctx context.Context, e string) (Observation, error) {
		calls++
		if d, ok := ctx.Deadline(); !ok || time.Until(d) > 3*time.Second {
			t.Fatal("unbounded lookup")
		}
		if e == "a.local:1" {
			return Observation{}, errors.New("private key path must not leak")
		}
		return o, nil
	}
	a, err := Inventory(context.Background(), hints, r.Cluster, lookup, func() time.Time { return selectionNow })
	if err != nil || len(a) != 2 || calls != 2 || a[0].Reason != "unavailable_or_unauthenticated" || a[1].Reason != "available" {
		t.Fatal(a, calls, err)
	}
	b, err := Inventory(context.Background(), hints, r.Cluster, lookup, func() time.Time { return selectionNow })
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal(b, err)
	}
	encoded, _ := json.Marshal(a)
	for _, bad := range []string{"forged", "private key"} {
		if strings.Contains(string(encoded), bad) {
			t.Fatal(string(encoded))
		}
	}
}

func TestInventoryIdentityFreshnessAndCancellation(t *testing.T) {
	for _, test := range []struct {
		name, want string
		mutate     func(*Observation)
	}{
		{"wrong-role", "wrong_identity", func(o *Observation) { o.Peer.Role = trust.Controller }},
		{"wrong-cluster", "wrong_identity", func(o *Observation) { o.Peer.Cluster = strings.Repeat("d", 32) }},
		{"expired", "wrong_identity", func(o *Observation) { o.Peer.ExpiresAt = selectionNow }},
		{"stale", "stale_capabilities", func(o *Observation) { o.Capabilities.ProbeObservedAt -= 121 }},
		{"unavailable", "not_recruitable", func(o *Observation) { o.Capabilities.Probe.MLXSupported = false; o.Capabilities.Recruitable = false }},
		{"invalid", "invalid_capabilities", func(o *Observation) { o.Capabilities.Version = "other" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			r, o := fixture()
			test.mutate(&o)
			rows, err := Inventory(context.Background(), []discovery.Hint{{Service: discovery.Node, Endpoint: "node.local:1"}}, r.Cluster, func(context.Context, string) (Observation, error) { return o, nil }, func() time.Time { return selectionNow })
			if err != nil || len(rows) != 1 || rows[0].Reason != test.want {
				t.Fatal(rows, err)
			}
			if (test.want == "wrong_identity" || test.want == "invalid_capabilities") && rows[0].Capabilities != nil {
				t.Fatal("published unverified capabilities")
			}
		})
	}
	r, _ := fixture()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Inventory(ctx, nil, r.Cluster, func(context.Context, string) (Observation, error) {
		t.Fatal("lookup after cancel")
		return Observation{}, nil
	}, time.Now); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
