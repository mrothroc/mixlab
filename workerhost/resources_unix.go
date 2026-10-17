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
const resourceSampleGrace = 6 * time.Second
const resourceEntryLimit = 100000

var errResourceSampleStale = errors.New("worker resource sample freshness expired")

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
		err := monitorResources(ctx, cancel, p, dir, limits, sample)
		if err != nil {
			cancel()
			err = fmt.Errorf("worker resource monitor: %w", err)
		}
		done <- err
	}()
	return func() error {
		stop()
		return <-done
	}
}

func monitorResources(ctx context.Context, cancel context.CancelFunc, p *ownedProcess, dir string, limits contract.Limits, sample resourceSampler) error {
	var meter resourceMeter
	timeouts := 0
	ticker := time.NewTicker(resourceInterval)
	defer ticker.Stop()
	graceCtx, stopGrace := resourceGrace(ctx, cancel)
	defer func() { stopGrace() }()
	stale := func() error {
		return fmt.Errorf("no healthy resource sample within %s: %w", resourceSampleGrace, context.DeadlineExceeded)
	}
	for {
		probeCtx, stopProbe := context.WithTimeout(graceCtx, resourceSampleTimeout)
		u, err := sample(probeCtx, p, dir)
		if err == nil {
			// Completed evidence remains authoritative even during shutdown.
			err = meter.check(u, limits)
		}
		if err == nil {
			err = probeCtx.Err()
		}
		stopProbe()
		if context.Cause(graceCtx) == errResourceSampleStale {
			if err == nil || resourceSampleInterrupted(err) {
				return stale()
			}
			return err
		}
		if ctx.Err() != nil && resourceSampleInterrupted(err) {
			return nil
		}
		if err != nil {
			// Do not use errors.As here: a joined permanent failure must not
			// become retryable just because it also contains a sampler timeout.
			if _, ok := err.(*processSampleTimeoutError); !ok || timeouts != 0 {
				return err
			}
			timeouts++
			// Keep both the CPU baseline and freshness deadline across the gap.
		} else {
			if stopGrace() {
				return stale()
			}
			graceCtx, stopGrace = resourceGrace(ctx, cancel)
			timeouts = 0
		}
		select {
		case <-ctx.Done():
			if context.Cause(graceCtx) == errResourceSampleStale {
				return stale()
			}
			return nil
		case <-ticker.C:
		}
	}
}

func resourceSampleInterrupted(err error) bool {
	switch err.(type) {
	case *processSampleTimeoutError, *processSampleCanceledError:
		return true
	default:
		return err == context.Canceled || err == context.DeadlineExceeded
	}
}

// The independent freshness deadline cancels supervision even if filesystem
// sampling stalls. Join its callback on reset so it cannot cancel a later window.
func resourceGrace(ctx context.Context, cancel context.CancelFunc) (context.Context, func() bool) {
	grace, stop := context.WithTimeoutCause(ctx, resourceSampleGrace, errResourceSampleStale)
	done := make(chan struct{})
	stopCallback := context.AfterFunc(grace, func() {
		if context.Cause(grace) == errResourceSampleStale {
			cancel()
		}
		close(done)
	})
	return grace, func() bool {
		stopped := stopCallback()
		stop()
		if stopped {
			close(done)
		} else {
			<-done
		}
		return context.Cause(grace) == errResourceSampleStale
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
