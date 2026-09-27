//go:build darwin || linux

package statehome

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"testing"
	"time"
)

func TestProcessLockSubprocess(t *testing.T) {
	if dir := os.Getenv("MIXLAB_TEST_PROCESS_LOCK"); dir != "" {
		p := resolved(t, Options{ExactDir: dir}, Context{Kind: Principal})
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := p.WithProcessLock(ctx, "counter.lock", func() error {
			b, err := p.ReadFile("counter")
			if err != nil {
				return err
			}
			n, err := strconv.Atoi(string(b))
			if err != nil {
				return err
			}
			time.Sleep(10 * time.Millisecond)
			return p.WriteFile("counter", []byte(strconv.Itoa(n+1)))
		}); err != nil {
			t.Fatal(err)
		}
		return
	}
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	if err := p.WriteFile("counter", []byte("0")); err != nil {
		t.Fatal(err)
	}
	var cmds []*exec.Cmd
	for i := 0; i < 6; i++ {
		cmd := exec.Command(os.Args[0], "-test.run=^TestProcessLockSubprocess$", "-test.timeout=20s")
		cmd.Env = append(os.Environ(), "MIXLAB_TEST_PROCESS_LOCK="+p.Dir())
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cmds = append(cmds, cmd)
	}
	for _, cmd := range cmds {
		if err := cmd.Wait(); err != nil {
			t.Error(err)
		}
	}
	b, err := p.ReadFile("counter")
	if err != nil || string(b) != "6" {
		t.Fatalf("lost updates %q: %v", b, err)
	}
}

func TestProcessLockCancellationAndFailure(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	entered, release := make(chan struct{}), make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- p.WithProcessLock(context.Background(), "operation.lock", func() error { close(entered); <-release; return nil })
	}()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	called := false
	err := p.WithProcessLock(ctx, "operation.lock", func() error { called = true; return nil })
	cancel()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !errors.Is(err, context.DeadlineExceeded) || called {
		t.Fatal("wait was not canceled", err)
	}
	want := errors.New("operation failure")
	if err := p.WithProcessLock(context.Background(), "operation.lock", func() error { return want }); !errors.Is(err, want) {
		t.Fatal(err)
	}
	if err := p.WithProcessLock(context.Background(), "operation.lock", func() error { return nil }); err != nil {
		t.Fatal("lock leaked", err)
	}
	if err := p.WriteFile("corrupt.lock", []byte("bad")); err != nil {
		t.Fatal(err)
	}
	if err := p.WithProcessLock(context.Background(), "corrupt.lock", func() error { t.Fatal("ran with corrupt marker"); return nil }); err == nil {
		t.Fatal("corrupt marker accepted")
	}
}
