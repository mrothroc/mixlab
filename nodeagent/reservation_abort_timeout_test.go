package nodeagent

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"
)

func TestReservationAbortLockTimeoutAndLostACKRetry(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	q := reservation("1")
	if _, err := s.Reserve(ctx, actor, q, leaseNow); err != nil {
		t.Fatal(err)
	}
	before, _, err := s.load()
	if err != nil {
		t.Fatal(err)
	}
	locked, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		done <- s.path.WithProcessLock(ctx, nodeLock, func() error {
			close(locked)
			<-release
			return nil
		})
	}()
	<-locked
	for range 2 {
		short, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
		_, err = s.AbortReservation(short, actor, q, leaseNow)
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("lock timeout: %v", err)
		}
	}
	close(release)
	if err = <-done; err != nil {
		t.Fatal(err)
	}
	after, _, err := s.load()
	if err != nil || !bytes.Equal(before, after) {
		t.Fatal("timed-out abort assumed release", err)
	}
	first, err := s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || first.Lease == nil || first.Lease.State != Releasing {
		t.Fatal("abort retry did not persist release intent", err)
	}
	// Treat the first response as lost, then reopen the store and retry exactly.
	s, err = Open(s.path, s.cluster, s.node)
	if err != nil {
		t.Fatal(err)
	}
	again, err := s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || again.Lease == nil || *again.Lease != *first.Lease {
		t.Fatal("lost abort ACK changed release intent", err)
	}
	if _, err = s.Reserve(ctx, actor, q, leaseNow); err == nil {
		t.Fatal("late reserve crossed release fence")
	}
	if err = s.ReleaseUnprepared(ctx, first.Lease.ID); err != nil {
		t.Fatal(err)
	}
	again, err = s.AbortReservation(ctx, actor, q, leaseNow)
	if err != nil || again.Lease == nil || again.Lease.State != Released {
		t.Fatal("abort did not observe confirmed cleanup", err)
	}
}
