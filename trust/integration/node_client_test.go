package integration

import (
	"context"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

func TestNodeClientLeaseRoundTripAndReconciledRelease(t *testing.T) {
	x := newNodeJobsFixture(t)
	h, err := clusterapp.NodeManagementHandler(x.store, x.policy, x.f.clock, x.app)
	check(t, err)
	s := serveCapabilities(t, x.policy, h)
	c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(s.URL, "https://"), x.options.Node, func() (*managedtls.Policy, error) { return x.client, nil })
	check(t, err)
	ctx := context.Background()
	q := nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}
	l, err := c.Reserve(ctx, q)
	check(t, err)
	again, err := c.Reserve(ctx, q)
	check(t, err)
	if l != again {
		t.Fatal("retry changed reservation")
	}
	got, err := c.LeaseStatus(ctx, l.ID)
	check(t, err)
	if got != l {
		t.Fatal(got, l)
	}
	_, err = c.Release(ctx, nodeagent.LeaseCommand{IdempotencyKey: id(t), Lease: l.ID, ExpectedVersion: l.Version})
	check(t, err)
	got, err = c.LeaseStatus(ctx, l.ID)
	check(t, err)
	if got.State != nodeagent.Releasing {
		t.Fatal("release intent incorrectly considered terminal", got)
	}
	check(t, x.store.ReleaseUnprepared(ctx, l.ID))
	got, err = c.LeaseStatus(ctx, l.ID)
	check(t, err)
	if got.State != nodeagent.Released {
		t.Fatal("terminal cleanup missing", got)
	}
}

func TestNodeClientPinsPeerBeforeMutatingRequest(t *testing.T) {
	x := newNodeJobsFixture(t)
	var reached atomic.Int32
	s := serveCapabilities(t, x.policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reached.Add(1); w.WriteHeader(http.StatusOK) }))
	c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(s.URL, "https://"), id(t), func() (*managedtls.Policy, error) { return x.client, nil })
	check(t, err)
	_, err = c.Reserve(context.Background(), nodeagent.Reserve{IdempotencyKey: id(t), Run: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, TTLSeconds: 300})
	if err == nil || reached.Load() != 0 {
		t.Fatal("wrong selected node received operation", err, reached.Load())
	}
}

func TestNodeClientReservationAbortFencesUnobservedRequest(t *testing.T) {
	x := newNodeJobsFixture(t)
	h, err := clusterapp.NodeManagementHandler(x.store, x.policy, x.f.clock, x.app)
	check(t, err)
	s := serveCapabilities(t, x.policy, h)
	c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(s.URL, "https://"), x.options.Node, func() (*managedtls.Policy, error) { return x.client, nil })
	check(t, err)
	q := nodeagent.Reserve{IdempotencyKey: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: id(t), TTLSeconds: 300}
	out, err := c.AbortReservation(context.Background(), q)
	check(t, err)
	if out.Lease != nil {
		t.Fatal("unattempted reservation allocated")
	}
	if _, err := c.Reserve(context.Background(), q); err == nil {
		t.Fatal("delayed HTTP reservation escaped abort")
	}
}

func TestNodeClientReauthenticatesAfterResponse(t *testing.T) {
	x := newNodeJobsFixture(t)
	h, err := clusterapp.NodeManagementHandler(x.store, x.policy, x.f.clock, x.app)
	check(t, err)
	s := serveCapabilities(t, x.policy, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.ServeHTTP(w, r)
		x.f.mu.Lock()
		x.f.snapshot.Payload.Generation++
		x.f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: x.options.Node, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
		x.f.signSnapshot(t)
		x.f.mu.Unlock()
	}))
	c, err := clusterapp.NewNodeClientWithPolicy(strings.TrimPrefix(s.URL, "https://"), x.options.Node, func() (*managedtls.Policy, error) { return x.client, nil })
	check(t, err)
	_, err = c.Reserve(context.Background(), nodeagent.Reserve{IdempotencyKey: id(t), Run: id(t), ExpectedNodeVersion: 1, CapabilityGeneration: 1, TTLSeconds: 300})
	if err == nil {
		t.Fatal("accepted response after peer revocation")
	}
}
