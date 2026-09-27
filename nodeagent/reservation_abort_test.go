package nodeagent

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestReservationAbortFullFenceHistoryStillReleasesExistingLease(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	q := reservation("1")
	l, err := s.Reserve(ctx, actor, q, leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	old, r, err := s.load()
	if err != nil {
		t.Fatal(err)
	}
	for i := range maxReceipts {
		r.ReservationFences = append(r.ReservationFences, reservationFence{actor.Principal, fmt.Sprintf("%032x", i), digest(q)})
	}
	if err := s.save(old, r); err != nil {
		t.Fatal(err)
	}
	f, err := s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || f.Lease == nil || f.Lease.State != Releasing {
		t.Fatal(f, err)
	}
	if err := s.ReleaseUnprepared(ctx, l.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Reserve(ctx, actor, q, leaseNow); err == nil {
		t.Fatal("terminal receipt replayed as live reservation")
	}
	f, err = s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || f.Lease == nil || f.Lease.State != Released {
		t.Fatal(f, err)
	}
}

func TestReservationAbortBeforeReserveIsPermanent(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	q := reservation("1")
	f, err := s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || f.Lease != nil || f.Node != s.node || f.Controller != actor.Principal || f.RequestHash != digest(q) {
		t.Fatal(f, err)
	}
	s, err = Open(s.path, s.cluster, s.node)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Reserve(ctx, actor, q, leaseNow); err == nil {
		t.Fatal("delayed reserve crossed abort fence")
	}
	again, err := s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || !reflect.DeepEqual(f, again) {
		t.Fatal(again, err)
	}
	changed := q
	changed.Run = strings.Repeat("9", 32)
	if _, err := s.AbortReservation(ctx, actor, changed, leaseNow); err == nil {
		t.Fatal("abort accepted changed request")
	}
	av, err := s.Availability(leaseNow)
	if err != nil || !av.Available {
		t.Fatal("no-op abort allocated lease", av, err)
	}
}

func TestReservationAbortRacesReserveAndPreservesOtherOwner(t *testing.T) {
	for range 10 {
		s, actor := leaseFixture(t)
		ctx := context.Background()
		q := reservation("1")
		var wg sync.WaitGroup
		wg.Add(2)
		go func() { defer wg.Done(); _, _ = s.Reserve(ctx, actor, q, leaseNow) }()
		go func() {
			defer wg.Done()
			_, err := s.AbortReservation(ctx, actor, q, leaseNow)
			if err != nil {
				t.Error(err)
			}
		}()
		wg.Wait()
		if _, err := s.Reserve(ctx, actor, q, leaseNow); err == nil {
			t.Fatal("revived fenced reservation")
		}
		f, err := s.AbortReservation(ctx, actor, q, leaseNow)
		if err != nil {
			t.Fatal(err)
		}
		if f.Lease != nil {
			if f.Lease.State != Releasing {
				t.Fatal("abort claims premature physical cleanup", f)
			}
			if err := s.ReleaseUnprepared(ctx, f.Lease.ID); err != nil {
				t.Fatal(err)
			}
			f, err = s.AbortReservation(ctx, actor, q, leaseNow)
			if err != nil || f.Lease.State != Released {
				t.Fatal(f, err)
			}
		}
	}
	s, actor := leaseFixture(t)
	ctx := context.Background()
	q := reservation("1")
	l, err := s.Reserve(ctx, actor, q, leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	other := actor
	other.Principal = strings.Repeat("8", 32)
	f, err := s.AbortReservation(ctx, other, q, leaseNow)
	if err != nil || f.Lease != nil {
		t.Fatal(f, err)
	}
	got, err := s.LeaseStatus(ctx, actor, l.ID, leaseNow)
	if err != nil || got != l {
		t.Fatal("foreign fence changed owner lease", got, err)
	}
	if _, err := s.LeaseStatus(ctx, other, l.ID, leaseNow); err == nil {
		t.Fatal("foreign controller read lease")
	}
}
