//go:build darwin || linux

package main

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerprobe"
)

type retrySignalLog struct {
	ready chan struct{}
	once  sync.Once
}

func (l *retrySignalLog) Write(b []byte) (int, error) {
	if strings.Contains(string(b), "retry in") {
		l.once.Do(func() { close(l.ready) })
	}
	return len(b), nil
}

func TestAgentServiceKeepsApprovalLockDuringRetry(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	log := &retrySignalLog{ready: make(chan struct{})}
	done := make(chan int, 1)
	go func() {
		// Missing installation metadata deliberately fails the first attempt.
		done <- runAgentSupervised(ctx, []string{"-agent-state-dir", dir}, io.Discard, log, func(attempt func() int) int {
			return runServiceRetry(ctx, log, attempt)
		})
	}()
	select {
	case <-log.ready:
	case <-ctx.Done():
		t.Fatal("service never entered retry")
	}
	short, stop := context.WithTimeout(ctx, 50*time.Millisecond)
	_, err = clusterapp.ReapproveNodeInstallation(short, p, "/unused/worker", "/unused/guardian", func(context.Context, string, string) (workerprobe.Report, error) {
		t.Error("probe ran while service supervisor was alive")
		return workerprobe.Report{}, nil
	})
	stop()
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("reapprove bypassed supervisor lock: %v", err)
	}
	cancel()
	if code := <-done; code != 0 {
		t.Fatalf("stop code=%d", code)
	}
	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := p.WithProcessLock(ctx, clusterapp.NodeServiceLock, func() error { return nil }); err != nil {
		t.Fatal("supervisor did not release approval lock", err)
	}
}
