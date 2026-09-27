package principal_test

import (
	"context"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestSnapshotReceiverNeverGrantsApplicationAuthority(t *testing.T) {
	f := setup(t, trust.Node)
	s := f.installed(t)
	stale := f.now.Add(trust.SnapshotLifetime + time.Second)
	if _, _, err := s.Active(stale); err == nil {
		t.Fatal("stale node active")
	}
	r, key, err := s.SnapshotReceiver(stale)
	check(t, err)
	if key == nil || r.Principal != f.state.Principal {
		t.Fatal("receiver identity changed")
	}
	check(t, s.Refresh(context.Background(), f.snapshot(t, stale, true), stale))
	if _, _, err := s.Active(stale); err == nil {
		t.Fatal("revoked node active")
	}
	_, _, err = s.SnapshotReceiver(stale)
	check(t, err)
	leaf, _, err := certificates.Parse(r.Chain[0])
	check(t, err)
	if _, _, err := s.SnapshotReceiver(leaf.NotAfter.Add(time.Second)); err == nil {
		t.Fatal("expired node recovered without enrollment")
	}
	other := setup(t, trust.Controller)
	if _, _, err := other.installed(t).SnapshotReceiver(other.now); err == nil {
		t.Fatal("controller became snapshot node")
	}
}
