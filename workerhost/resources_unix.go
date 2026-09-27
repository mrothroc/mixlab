//go:build darwin || linux

package workerhost

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/workerhost/contract"
)

const resourceInterval = time.Second
const resourceSampleTimeout = 2 * time.Second
const resourceEntryLimit = 100000

// ResourceLimitError identifies a failed monitored budget, not an OS quota.
type ResourceLimitError struct {
	Resource string
	Observed int64
	Limit    int64
}

func (e *ResourceLimitError) Error() string {
	return fmt.Sprintf("worker %s limit exceeded: observed=%d limit=%d", e.Resource, e.Observed, e.Limit)
}

type processUsage struct {
	Identity string
	CPU      time.Duration
	RSS      int64
}

type resourceUsage struct {
	Processes []processUsage
	Disk      int64
}

type resourceSampler func(context.Context, *ownedProcess, string) (resourceUsage, error)

// Retain consumed CPU when a sampled child disappears. Track only the current
// snapshot so process churn cannot grow an unbounded PID history. Unsampled
// short-lived children are a documented limitation of monitored accounting.
type resourceMeter struct {
	previous map[string]time.Duration
	cpu      time.Duration
}

func (m *resourceMeter) check(u resourceUsage, l contract.Limits) error {
	next := make(map[string]time.Duration, len(u.Processes))
	var rss int64
	for _, p := range u.Processes {
		if p.Identity == "" || p.CPU < 0 || p.RSS < 0 {
			return fmt.Errorf("invalid worker resource sample")
		}
		if _, exists := next[p.Identity]; exists {
			return fmt.Errorf("duplicate worker process sample")
		}
		delta := p.CPU - m.previous[p.Identity]
		if delta < 0 {
			// A reused PID/start-second or a platform counter reset is not a
			// refund. Count the observed new lifetime instead.
			delta = p.CPU
		}
		next[p.Identity] = p.CPU
		budget := time.Duration(l.CPUSeconds) * time.Second
		if delta > budget-m.cpu {
			return &ResourceLimitError{"cpu_nanoseconds", saturatedSum(int64(m.cpu), int64(delta)), int64(budget)}
		}
		m.cpu += delta
		if p.RSS > l.MemoryBytes-rss {
			return &ResourceLimitError{"memory_bytes", saturatedSum(rss, p.RSS), l.MemoryBytes}
		}
		rss += p.RSS
	}
	m.previous = next
	if u.Disk < 0 {
		return fmt.Errorf("invalid worker disk sample")
	}
	if u.Disk > l.DiskBytes {
		return &ResourceLimitError{"disk_bytes", u.Disk, l.DiskBytes}
	}
	return nil
}

func saturatedSum(a, b int64) int64 {
	if b > (1<<63-1)-a {
		return 1<<63 - 1
	}
	return a + b
}

// stop joins the only sampler before returning its failure. All error paths
// cancel the same supervision context; physical process cleanup remains owned
// by Supervisor.Run, never by a sampling goroutine.
func watchResources(parent context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler) func() error {
	if sample == nil {
		sample = sampleResources
	}
	ctx, stop := context.WithCancel(parent)
	done := make(chan error, 1)
	go func() {
		var meter resourceMeter
		ticker := time.NewTicker(resourceInterval)
		defer ticker.Stop()
		for {
			probeCtx, stopProbe := context.WithTimeout(ctx, resourceSampleTimeout)
			u, err := sample(probeCtx, p, dir)
			if err == nil {
				// A completed observation is evidence even when shutdown races
				// its return. Never erase a known breach on the success path.
				err = meter.check(u, limits)
			}
			if err == nil {
				err = probeCtx.Err()
			}
			stopProbe()
			if ctx.Err() != nil && (errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)) {
				done <- nil
				return
			}
			if err != nil {
				cancel()
				done <- fmt.Errorf("worker resource monitor: %w", err)
				return
			}
			select {
			case <-ctx.Done():
				done <- nil
				return
			case <-ticker.C:
			}
		}
	}()
	return func() error {
		stop()
		return <-done
	}
}

func sampleResources(ctx context.Context, p *ownedProcess, dir string) (resourceUsage, error) {
	processes, err := p.sampleUsage(ctx)
	if err != nil {
		return resourceUsage{}, err
	}
	disk, err := attemptDiskBytes(ctx, dir)
	return resourceUsage{Processes: processes, Disk: disk}, err
}

func checkAttemptDisk(parent context.Context, dir string, limit int64) error {
	ctx, cancel := context.WithTimeout(parent, resourceSampleTimeout)
	defer cancel()
	n, err := attemptDiskBytes(ctx, dir)
	if err != nil {
		return fmt.Errorf("worker disk monitor: %w", err)
	}
	if n > limit {
		return &ResourceLimitError{"disk_bytes", n, limit}
	}
	return nil
}

func attemptDiskBytes(ctx context.Context, dir string) (int64, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return 0, err
	}
	defer func() { _ = root.Close() }()
	var total int64
	entries := 0
	var walk func(string, int) error
	walk = func(path string, depth int) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if depth > 128 {
			return fmt.Errorf("worker disk sample exceeds 128 directory levels")
		}
		// Root prevents escape through a concurrent symlink replacement.
		// O_DIRECTORY also prevents opening a substituted FIFO/device.
		f, err := root.OpenFile(path, os.O_RDONLY|syscall.O_DIRECTORY, 0)
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		for {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			batch, readErr := f.ReadDir(128)
			if readErr != nil && !errors.Is(readErr, io.EOF) {
				return readErr
			}
			for _, entry := range batch {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				entries++
				if entries > resourceEntryLimit {
					return fmt.Errorf("worker disk sample exceeds %d entries", resourceEntryLimit)
				}
				name := filepath.Join(path, entry.Name())
				info, err := root.Lstat(name)
				if errors.Is(err, os.ErrNotExist) {
					continue
				}
				if err != nil {
					return err
				}
				if info.IsDir() {
					if err := walk(name, depth+1); err != nil {
						return err
					}
				} else {
					if info.Size() < 0 || info.Size() > (1<<63-1)-total {
						return fmt.Errorf("worker disk sample size overflow")
					}
					total += info.Size()
				}
			}
			if errors.Is(readErr, io.EOF) {
				return nil
			}
		}
	}
	err = walk(".", 0)
	if err == nil {
		err = ctx.Err()
	}
	return total, err
}
