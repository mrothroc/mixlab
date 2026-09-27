package nodeagent

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestRefreshIdleProbePreservesApprovedContract(t *testing.T) {
	s, actor := leaseFixture(t)
	p := profileFixture(s)
	total := uint64(1 << 30)
	p.Probe.MemoryBytes = &total
	ctx := context.Background()
	if err := s.InstallProfile(ctx, p); err != nil {
		t.Fatal(err)
	}
	now := leaseNow.Add(time.Minute)
	free := uint64(128 << 20)
	probe := p.Probe
	probe.FreeMemoryBytes = &free
	if err := s.RefreshIdleProbe(ctx, func(context.Context) (workerprobe.Report, error) { return probe, nil }, func() time.Time { return now }); err != nil {
		t.Fatal(err)
	}
	c, err := s.Capabilities(ctx, actor, now)
	if err != nil || c.ProbeObservedAt != now.Unix() || c.Generation != p.Generation || !reflect.DeepEqual(c.Probe, probe) {
		t.Fatal(c, err)
	}
	for _, mutate := range []func(*workerprobe.Report){
		func(r *workerprobe.Report) { r.DeviceName = "different GPU" },
		func(r *workerprobe.Report) { r.BuildID = "changed" },
	} {
		changed := probe
		mutate(&changed)
		if err := s.RefreshIdleProbe(ctx, func(context.Context) (workerprobe.Report, error) { return changed, nil }, func() time.Time { return now.Add(time.Minute) }); err == nil {
			t.Fatal("accepted changed probe contract")
		}
	}
	after, err := s.Capabilities(ctx, actor, now)
	if err != nil || !reflect.DeepEqual(c, after) {
		t.Fatal("failed probe mutated profile", after, err)
	}
}

func TestRefreshIdleProbeSkipsOwnedAcceleratorAndCancellation(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	if err := s.InstallProfile(ctx, profileFixture(s)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Reserve(ctx, actor, reservation("1"), leaseNow); err != nil {
		t.Fatal(err)
	}
	called := false
	probe := func(context.Context) (workerprobe.Report, error) {
		called = true
		return workerprobe.Report{}, errors.New("must not run")
	}
	if err := s.RefreshIdleProbe(ctx, probe, func() time.Time { return leaseNow }); err != nil || called {
		t.Fatal("leased probe ran", err)
	}
	if err := s.Expire(ctx, leaseNow.Add(61*time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := s.RefreshIdleProbe(ctx, probe, func() time.Time { return leaseNow }); err != nil || called {
		t.Fatal("releasing probe ran", err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := s.RefreshIdleProbe(canceled, probe, func() time.Time { return leaseNow }); !errors.Is(err, context.Canceled) || called {
		t.Fatal("canceled probe ran", err)
	}
}
