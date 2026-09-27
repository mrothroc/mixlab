package nodeagent

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

var leaseNow = time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC)

func leaseFixture(t *testing.T) (*Store, trust.AuthenticatedPrincipal) {
	t.Helper()
	dir, e := filepath.EvalSymlinks(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	if e = os.Chmod(dir, 0700); e != nil {
		t.Fatal(e)
	}
	p, e := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	if e != nil {
		t.Fatal(e)
	}
	cluster, node := strings.Repeat("a", 32), strings.Repeat("b", 32)
	s, e := Initialize(context.Background(), p, cluster, node, 1)
	if e != nil {
		t.Fatal(e)
	}
	return s, trust.AuthenticatedPrincipal{Cluster: cluster, Principal: strings.Repeat("c", 32), Role: trust.Controller, NotBefore: leaseNow.Add(-time.Hour), ExpiresAt: leaseNow.Add(time.Hour)}
}
func reservation(key string) Reserve {
	return Reserve{IdempotencyKey: strings.Repeat(key, 32), ExpectedNodeVersion: 1, CapabilityGeneration: 1, Run: strings.Repeat("d", 32), TTLSeconds: 60}
}

func TestLeaseExclusiveAndDurableRetry(t *testing.T) {
	ctx := context.Background()
	s, actor := leaseFixture(t)
	var wg sync.WaitGroup
	leases := make(chan Lease, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			other, e := Open(s.path, s.cluster, s.node)
			if e != nil {
				t.Error(e)
				return
			}
			l, e := other.Reserve(ctx, actor, reservation("1"), leaseNow)
			if e != nil {
				t.Error(e)
				return
			}
			leases <- l
		}()
	}
	wg.Wait()
	close(leases)
	var first Lease
	for l := range leases {
		if first.ID == "" {
			first = l
		} else if !reflect.DeepEqual(first, l) {
			t.Fatal("concurrent retry created another lease")
		}
	}
	if first.ID == "" {
		t.Fatal("no lease")
	}
	competitor := actor
	competitor.Principal = strings.Repeat("e", 32)
	if _, e := s.Reserve(ctx, competitor, reservation("2"), leaseNow); e == nil {
		t.Fatal("another controller reserved busy node")
	}
	changed := reservation("1")
	changed.TTLSeconds = 90
	if _, e := s.Reserve(ctx, actor, changed, leaseNow); e == nil {
		t.Fatal("idempotency mismatch accepted")
	}
	if e := s.InstallProfile(ctx, profileFixture(s)); e == nil {
		t.Fatal("busy capability changed")
	}
	a, e := s.Availability(leaseNow)
	if e != nil || a.Available || a.Lease.ID != first.ID {
		t.Fatal(a, e)
	}
}

func TestMissingLeaseHistoryCannotBeReinitialized(t *testing.T) {
	s, _ := leaseFixture(t)
	if err := os.Remove(filepath.Join(s.path.Dir(), leaseFile)); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(s.path, s.cluster, s.node); err == nil {
		t.Fatal("opened missing history")
	}
	if _, err := Initialize(context.Background(), s.path, s.cluster, s.node, 1); err == nil {
		t.Fatal("reset published lease history")
	}
}

func TestNodeInitializationReadyBoundary(t *testing.T) {
	for _, point := range []string{"identity", "journal"} {
		t.Run(point, func(t *testing.T) {
			s, _ := leaseFixture(t)
			if err := os.Remove(filepath.Join(s.path.Dir(), nodeReadyFile)); err != nil {
				t.Fatal(err)
			}
			if point == "identity" {
				if err := os.Remove(filepath.Join(s.path.Dir(), leaseFile)); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := Open(s.path, s.cluster, s.node); err == nil {
				t.Fatal("opened incomplete initialization")
			}
			if _, err := Initialize(context.Background(), s.path, s.cluster, s.node, 1); err != nil {
				t.Fatal(err)
			}
			if _, err := Open(s.path, s.cluster, s.node); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestReleasePreservesExpiryReason(t *testing.T) {
	ctx := context.Background()
	s, actor := leaseFixture(t)
	l, err := s.Reserve(ctx, actor, reservation("1"), leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Expire(ctx, leaseNow.Add(61*time.Second)); err != nil {
		t.Fatal(err)
	}
	a, err := s.Availability(leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	q := LeaseCommand{IdempotencyKey: strings.Repeat("2", 32), Lease: l.ID, ExpectedVersion: a.Lease.Version}
	got, err := s.ReleaseLease(ctx, actor, q, leaseNow.Add(61*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if got.CleanupReason != "expired" || got.Version != a.Lease.Version {
		t.Fatal("release rewrote cleanup intent", got)
	}
}

func TestLeaseRenewReleaseAndExpiry(t *testing.T) {
	ctx := context.Background()
	s, actor := leaseFixture(t)
	l, e := s.Reserve(ctx, actor, reservation("1"), leaseNow)
	if e != nil {
		t.Fatal(e)
	}
	q := LeaseCommand{IdempotencyKey: strings.Repeat("2", 32), Lease: l.ID, ExpectedVersion: l.Version, TTLSeconds: 60}
	l, e = s.RenewLease(ctx, actor, q, leaseNow.Add(30*time.Second))
	if e != nil || l.Expires != leaseNow.Add(90*time.Second).Unix() {
		t.Fatal(l, e)
	}
	duplicate, e := s.RenewLease(ctx, actor, q, leaseNow.Add(31*time.Second))
	if e != nil || !reflect.DeepEqual(l, duplicate) {
		t.Fatal("renew retry changed outcome", duplicate, e)
	}
	q = LeaseCommand{IdempotencyKey: strings.Repeat("3", 32), Lease: l.ID, ExpectedVersion: l.Version}
	wrong := actor
	wrong.Principal = strings.Repeat("e", 32)
	if _, e := s.ReleaseLease(ctx, wrong, q, leaseNow); e == nil {
		t.Fatal("other controller released lease")
	}
	l, e = s.ReleaseLease(ctx, actor, q, leaseNow.Add(32*time.Second))
	if e != nil || l.State != Releasing {
		t.Fatal(l, e)
	}
	if a, e := s.Availability(leaseNow); e != nil || a.Available {
		t.Fatal("release intent freed node", a, e)
	}
	s, e = Open(s.path, s.cluster, s.node)
	if e != nil {
		t.Fatal(e)
	}
	if a, e := s.Availability(leaseNow); e != nil || a.Available {
		t.Fatal("restart freed uncertain lease", a, e)
	}
	if e := s.ReleaseUnprepared(ctx, l.ID); e != nil {
		t.Fatal(e)
	}
	a, e := s.Availability(leaseNow)
	if e != nil || !a.Available {
		t.Fatal(a, e)
	}
	if e := s.ReleaseUnprepared(ctx, l.ID); e != nil {
		t.Fatal("terminal retry", e)
	}
	r := reservation("4")
	r.ExpectedNodeVersion = a.NodeVersion
	l, e = s.Reserve(ctx, actor, r, leaseNow.Add(40*time.Second))
	if e != nil {
		t.Fatal(e)
	}
	if e := s.Expire(ctx, leaseNow.Add(101*time.Second)); e != nil {
		t.Fatal(e)
	}
	a, e = s.Availability(leaseNow.Add(101 * time.Second))
	if e != nil || a.Available || a.Lease.State != Releasing || a.Lease.CleanupReason != "expired" {
		t.Fatal(a, e)
	}
	q = LeaseCommand{IdempotencyKey: strings.Repeat("5", 32), Lease: l.ID, ExpectedVersion: a.Lease.Version, TTLSeconds: 60}
	if _, e := s.RenewLease(ctx, actor, q, leaseNow.Add(101*time.Second)); e == nil {
		t.Fatal("expired lease revived")
	}
}

func TestLeaseValidationAndMissingHistory(t *testing.T) {
	ctx := context.Background()
	s, actor := leaseFixture(t)
	for _, mutate := range []func(*trust.AuthenticatedPrincipal){func(p *trust.AuthenticatedPrincipal) { p.Role = trust.Node }, func(p *trust.AuthenticatedPrincipal) { p.Cluster = strings.Repeat("e", 32) }, func(p *trust.AuthenticatedPrincipal) { p.ExpiresAt = leaseNow }} {
		p := actor
		mutate(&p)
		if _, e := s.Reserve(ctx, p, reservation("1"), leaseNow); e == nil {
			t.Fatal("invalid actor accepted")
		}
	}
	q := reservation("1")
	q.CapabilityGeneration = 2
	if _, e := s.Reserve(ctx, actor, q, leaseNow); e == nil {
		t.Fatal("stale capability accepted")
	}
	if e := os.Remove(filepath.Join(s.path.Dir(), leaseFile)); e != nil {
		t.Fatal(e)
	}
	if _, e := Open(s.path, s.cluster, s.node); e == nil {
		t.Fatal("missing lease history recreated")
	}
}
