//go:build darwin || linux

package statehome

import (
	"bytes"
	"sync"
	"testing"
	"time"
)

func TestProtectedReadsSerializeWithAtomicReplacement(t *testing.T) {
	for _, bounded := range []bool{false, true} {
		p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
		if err := p.WriteFile("state", []byte("old")); err != nil {
			t.Fatal(err)
		}
		dir, err := p.lock()
		if err != nil {
			t.Fatal(err)
		}
		type result struct {
			b   []byte
			err error
		}
		started := make(chan struct{})
		done := make(chan result, 1)
		go func() {
			close(started)
			var b []byte
			var err error
			if bounded {
				b, err = p.ReadFileLimit("state", 16)
			} else {
				b, err = p.ReadFile("state")
			}
			done <- result{b, err}
		}()
		<-started
		select {
		case got := <-done:
			release(dir)
			t.Fatalf("reader bypassed writer lock: %q %v", got.b, got.err)
		case <-time.After(20 * time.Millisecond):
		}
		err = p.writeFileLocked(dir, "state", []byte("new"))
		release(dir)
		if err != nil {
			t.Fatal(err)
		}
		got := <-done
		if got.err != nil || string(got.b) != "new" {
			t.Fatal(string(got.b), got.err)
		}
	}
}

func TestProtectedReadWriteConcurrency(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	value := bytes.Repeat([]byte("a"), 4096)
	if err := p.WriteFile("state", value); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for worker := range 6 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				if worker == 0 {
					if err := p.WriteFile("state", value); err != nil {
						t.Error(err)
						return
					}
				} else {
					b, err := p.ReadFileLimit("state", 4096)
					if err != nil || !bytes.Equal(b, value) {
						t.Errorf("atomic read: %v", err)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}
