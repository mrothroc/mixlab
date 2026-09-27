//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

func TestResourceCPUTimeParsing(t *testing.T) {
	for input, want := range map[string]time.Duration{
		"0:00.02": 20 * time.Millisecond, "139:12.34": 139*time.Minute + 12340*time.Millisecond,
		"02:03:04":   2*time.Hour + 3*time.Minute + 4*time.Second,
		"3-04:05:06": 76*time.Hour + 5*time.Minute + 6*time.Second,
	} {
		got, err := parseCPUTime(input)
		if err != nil || got != want {
			t.Fatalf("%q: %s %v, want %s", input, got, err, want)
		}
	}
	for _, s := range []string{"", "nan", "-1:00", "1:60", "1h:00", "1.5:00", "9999999999999:00", "999999-01:00:00"} {
		if _, err := parseCPUTime(s); err == nil {
			t.Fatalf("accepted %q", s)
		}
	}
}

func TestResourceProcessSnapshot(t *testing.T) {
	const snapshot = "10 10 0:01.20 100 Sat Sep 26 13:23:20 2026\n11 10 00:00:02 200 Sat Sep 26 13:23:21 2026\n12 12 0:10 900 Sat Sep 26 13:23:21 2026\n"
	p, err := parseProcessUsage(snapshot, 10)
	if err != nil || len(p) != 2 || p[0].CPU != 1200*time.Millisecond || p[1].RSS != 200*1024 || p[0].Identity == p[1].Identity {
		t.Fatalf("snapshot: %+v %v", p, err)
	}
	for _, bad := range []string{"", "10 10", strings.Replace(snapshot, "100 Sat", "-1 Sat", 1), strings.Replace(snapshot, "Sat Sep", "Sat Bad", 1)} {
		if _, err := parseProcessUsage(bad, 10); err == nil {
			t.Fatalf("accepted bad snapshot %q", bad)
		}
	}
	// io.Copy must not bypass the size check through a promoted ReadFrom method.
	b := &sampleBuffer{}
	if _, err := io.Copy(b, strings.NewReader(strings.Repeat("x", 8<<20+1))); err == nil || b.buffer.Len() > 8<<20 {
		t.Fatal("unbounded process output buffer")
	}
}

func TestResourceMeter(t *testing.T) {
	l := contract.Limits{CPUSeconds: 4, MemoryBytes: 1000, DiskBytes: 2000}
	var m resourceMeter
	for _, u := range []resourceUsage{
		{Processes: []processUsage{{"a", time.Second, 200}, {"b", time.Second, 300}}},
		{Processes: []processUsage{{"a", 2 * time.Second, 400}}},
		{Processes: []processUsage{{"c", time.Second, 400}}},
	} {
		if err := m.check(u, l); err != nil {
			t.Fatal(err)
		}
	}
	if m.cpu != 4*time.Second || len(m.previous) != 1 {
		t.Fatalf("departed CPU lost or unbounded history: %+v", m)
	}
	var limit *ResourceLimitError
	if err := m.check(resourceUsage{Processes: []processUsage{{"c", 2 * time.Second, 1}}}, l); !errors.As(err, &limit) || limit.Resource != "cpu_nanoseconds" || limit.Observed != int64(5*time.Second) {
		t.Fatalf("CPU limit: %v", err)
	}
	for _, tc := range []struct {
		u    resourceUsage
		kind string
	}{
		{resourceUsage{Processes: []processUsage{{"a", 0, 600}, {"b", 0, 500}}}, "memory_bytes"},
		{resourceUsage{Disk: 2001}, "disk_bytes"},
	} {
		m := resourceMeter{}
		if err := m.check(tc.u, l); !errors.As(err, &limit) || limit.Resource != tc.kind {
			t.Fatalf("%s: %v", tc.kind, err)
		}
	}
}

func TestResourceDiskSample(t *testing.T) {
	dir, external := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "file"), []byte("12345"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(external, "outside"), make([]byte, 1<<20), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(external, filepath.Join(dir, "link")); err != nil {
		t.Fatal(err)
	}
	n, err := attemptDiskBytes(context.Background(), dir)
	if err != nil || n != int64(5+len(external)) {
		t.Fatalf("followed external link or lost size: %d %v", n, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := attemptDiskBytes(ctx, dir); !errors.Is(err, context.Canceled) {
		t.Fatalf("ignored cancellation: %v", err)
	}
	if _, err := attemptDiskBytes(context.Background(), filepath.Join(dir, "missing")); err == nil {
		t.Fatal("accepted missing attempt")
	}
}

func TestSupervisedResourceLimits(t *testing.T) {
	for _, tc := range []struct{ mode, resource string }{
		{"resource-cpu", "cpu_nanoseconds"}, {"resource-memory", "memory_bytes"},
		{"resource-disk", "disk_bytes"}, {"resource-final-disk", "disk_bytes"},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			s, p := plan(t, tc.mode)
			p.StartupTimeout = 4 * time.Second
			switch tc.resource {
			case "cpu_nanoseconds":
				p.Limits.CPUSeconds = 1
			case "memory_bytes":
				p.Limits.MemoryBytes = 32 << 20
			case "disk_bytes":
				p.Limits.DiskBytes = 1 << 20
			}
			r, err := s.Run(context.Background(), p)
			var limit *ResourceLimitError
			if !errors.As(err, &limit) || limit.Resource != tc.resource {
				t.Fatalf("expected %s failure, result=%+v err=%v", tc.resource, r, err)
			}
			if r.PID == 0 || syscall.Kill(r.PID, 0) != syscall.ESRCH {
				t.Fatal("budget failure did not reap child")
			}
		})
	}
}

func TestSupervisedResourceSampleFailure(t *testing.T) {
	s, p := plan(t, "runtime-stall")
	s.sample = func(context.Context, *ownedProcess, string) (resourceUsage, error) {
		return resourceUsage{}, errors.New("intentional sampler failure")
	}
	r, err := s.Run(context.Background(), p)
	if err == nil || !strings.Contains(err.Error(), "intentional sampler failure") || r.PID == 0 || syscall.Kill(r.PID, 0) != syscall.ESRCH {
		t.Fatalf("sampler failure was not fatal/reaped: %+v %v", r, err)
	}
}

func TestSupervisedResourceLimitsRequired(t *testing.T) {
	s, p := plan(t, "success")
	p.Limits.MemoryBytes = 0
	if r, err := s.Run(context.Background(), p); err == nil || r.PID != 0 {
		t.Fatalf("missing limit launched: %+v %v", r, err)
	}
}

func TestResourceShutdownRetainsCompletedOverage(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	sample := func(ctx context.Context, _ *ownedProcess, _ string) (resourceUsage, error) {
		close(started)
		<-ctx.Done()
		return resourceUsage{Processes: []processUsage{{"worker", 0, 101}}}, nil
	}
	stop := watchResources(ctx, cancel, nil, "", contract.Limits{CPUSeconds: 1, MemoryBytes: 100, DiskBytes: 100}, sample)
	<-started
	var limit *ResourceLimitError
	if err := stop(); !errors.As(err, &limit) || limit.Resource != "memory_bytes" {
		t.Fatalf("discarded completed sample at shutdown: %v", err)
	}
}

func TestResourceSamplerCannotDelayPhysicalTermination(t *testing.T) {
	s, p := plan(t, "runtime-stall")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered := make(chan *ownedProcess, 1)
	release := make(chan struct{})
	s.sample = func(_ context.Context, process *ownedProcess, _ string) (resourceUsage, error) {
		entered <- process
		<-release // Model a filesystem read that cannot be interrupted.
		return resourceUsage{}, nil
	}
	done := make(chan error, 1)
	go func() { _, err := s.Run(ctx, p); done <- err }()
	var process *ownedProcess
	select {
	case process = <-entered:
	case <-time.After(5 * time.Second):
		close(release)
		t.Fatal("sampler did not start")
	}
	cancel()
	// Wait independently of the sampler. With the old defer order the child
	// remained live until release, despite supervision already being canceled.
	deadline := time.Now().Add(4 * time.Second)
	reaped := false
	for time.Now().Before(deadline) {
		process.mu.Lock()
		reaped = process.reaped
		process.mu.Unlock()
		if reaped {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	close(release)
	err := <-done
	if !reaped || err == nil {
		t.Fatalf("termination waited for stalled sample: reaped=%v err=%v", reaped, err)
	}
}
