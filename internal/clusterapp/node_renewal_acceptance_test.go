package clusterapp

import (
	"bytes"
	"context"
	"net"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestNodeRemoteRenewalAndExpiredReenrollment(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	p := initializedAt(t, endpoint)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { check(t, a.Close()) }()
	var seconds atomic.Int64
	seconds.Store(testNow.Unix())
	clock := func() time.Time { return time.Unix(seconds.Load(), 0) }
	done := make(chan error, 1)
	go func() { done <- ServeAuthority(ctx, l, a, clock) }()
	defer func() { cancel(); check(t, <-done) }()
	install := func(name string) (statehome.Path, *principal.Store) {
		t.Helper()
		resolve := func(suffix string, kind statehome.Kind) statehome.Path {
			out, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), name+suffix)}, statehome.Context{Kind: kind})
			check(t, err)
			return out
		}
		view, err := a.Current(ctx, clock())
		check(t, err)
		invite, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, view, clock())
		check(t, err)
		defer invite.Clear()
		path := resolve("", statehome.Principal)
		candidate, err := enrollee.Prepare(ctx, resolve("-stage", statehome.Enrollment), path, a.Anchor, trust.Node, "file", clock())
		check(t, err)
		defer func() { check(t, candidate.Close()) }()
		key, envelope, err := candidate.Keys()
		check(t, err)
		request, err := enrollment.NewRequest(invite, key, envelope, clock())
		check(t, err)
		result, err := a.Enrollment.Consume(ctx, request, invite.Secret, view, clock())
		check(t, err)
		check(t, candidate.CompleteProvisioned(ctx, invite, request, result, clock()))
		node, err := principal.Open(path, clock())
		check(t, err)
		t.Cleanup(func() { check(t, node.Close()) })
		return path, node
	}
	path, node := install("node")
	original, _, err := node.Active(clock())
	check(t, err)
	renew := func(ctx context.Context, _ time.Time) error { return RenewRemotePrincipal(ctx, node, endpoint, clock) }
	scheduler, err := PrincipalScheduler(path, node, renew)
	check(t, err)
	seconds.Store(testNow.Add(20 * 24 * time.Hour).Unix())
	_, err = a.Current(ctx, clock())
	check(t, err)
	check(t, RenewLocalPrincipal(ctx, a, a.Principal, clock()))
	check(t, RefreshNodeStartup(ctx, node, clock))
	state, err := scheduler.Tick(ctx, clock())
	check(t, err)
	if state.Outcome != "renewed" {
		t.Fatal("node scheduler did not renew", state)
	}
	seconds.Store(testNow.Add(31 * 24 * time.Hour).Unix())
	check(t, RefreshNodeStartup(ctx, node, clock))
	renewed, _, err := node.Active(clock())
	check(t, err)
	if renewed.Principal != original.Principal || renewed.Key.ID != original.Key.ID || bytes.Equal(renewed.Chain[0], original.Chain[0]) {
		t.Fatal("remote renewal changed identity or retained expired leaf")
	}
	// Keep the authority healthy while deliberately missing the node's renewal.
	seconds.Store(testNow.Add(40 * 24 * time.Hour).Unix())
	_, err = a.Current(ctx, clock())
	check(t, err)
	check(t, RenewLocalPrincipal(ctx, a, a.Principal, clock()))
	seconds.Store(testNow.Add(51 * 24 * time.Hour).Unix())
	if err := RefreshNodeStartup(ctx, node, clock); err == nil {
		t.Fatal("expired node recovered without enrollment")
	}
	scheduler, err = PrincipalScheduler(path, node, renew)
	check(t, err)
	if _, err := scheduler.Tick(ctx, clock()); err == nil {
		t.Fatal("restarted scheduler renewed expired identity")
	}
	_, replacement := install("replacement-node")
	next, _, err := replacement.Active(clock())
	check(t, err)
	if next.Principal == original.Principal || next.Key.ID == original.Key.ID {
		t.Fatal("reenrollment reused expired identity")
	}
	if _, _, err := node.Active(clock()); err == nil {
		t.Fatal("replacement enrollment revived old principal")
	}
}
