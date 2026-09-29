package nodeagent

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerprobe"
)

func TestReapprovePreservesHistoryAndRecoversInterruptedPublication(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	p := profileFixture(s)
	if err := s.InstallProfile(ctx, p); err != nil {
		t.Fatal(err)
	}
	report := p.Probe
	report.BuildID = strings.Repeat("f", 64)
	probe := func(context.Context) (workerprobe.Report, error) { return report, nil }
	broken := errors.New("publication interrupted")
	if err := s.ReapproveWorker(ctx, probe, func() error { return broken }, leaseNow); !errors.Is(err, broken) {
		t.Fatal(err)
	}
	if _, err := s.Capabilities(ctx, actor, leaseNow); err == nil {
		t.Fatal("uncommitted approval became recruitable")
	}
	if s.CheckApprovedWorker(report.BuildID) == nil {
		t.Fatal("interrupted approval passed startup check")
	}
	if err := s.ReapproveWorker(ctx, probe, func() error { return nil }, leaseNow); err != nil {
		t.Fatal(err)
	}
	c, err := s.Capabilities(ctx, actor, leaseNow)
	if err != nil {
		t.Fatal(err)
	}
	if c.Generation != 2 || c.Probe.BuildID != report.BuildID || c.Limits != p.Limits || c.DisplayName != p.DisplayName {
		t.Fatal(c)
	}
	_, after, err := s.loadProfile()
	if err != nil || !reflect.DeepEqual(p.Datasets, after.Datasets) {
		t.Fatal(after, err)
	}
	_, journal, err := s.load()
	if err != nil || len(journal.Leases) != 0 || journal.NodeVersion != 2 {
		t.Fatal(journal, err)
	}
}

func TestReapproveRejectsLeasesBeforeProbe(t *testing.T) {
	s, actor := leaseFixture(t)
	ctx := context.Background()
	if err := s.InstallProfile(ctx, profileFixture(s)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Reserve(ctx, actor, reservation("1"), leaseNow); err != nil {
		t.Fatal(err)
	}
	if err := s.ReapproveWorker(ctx, func(context.Context) (workerprobe.Report, error) {
		t.Fatal("probe while leased")
		return workerprobe.Report{}, nil
	}, func() error { t.Fatal("published while leased"); return nil }, leaseNow); err == nil {
		t.Fatal("leased approval")
	}
}

func TestReapproveSerializesReservations(t *testing.T) {
	s, actor := leaseFixture(t)
	p := profileFixture(s)
	if err := s.InstallProfile(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- s.ReapproveWorker(context.Background(), func(context.Context) (workerprobe.Report, error) { close(entered); <-release; return p.Probe, nil }, func() error { return nil }, leaseNow)
	}()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	_, err := s.Reserve(ctx, actor, reservation("1"), leaseNow)
	cancel()
	close(release)
	if approvalErr := <-done; approvalErr != nil {
		t.Fatal(approvalErr)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal("reservation raced approval", err)
	}
	if _, err := s.Reserve(context.Background(), actor, reservation("2"), leaseNow); err == nil {
		t.Fatal("stale capability generation reserved upgraded node")
	}
}
