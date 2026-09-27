//go:build darwin || linux

package statehome

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestCompareAndSwap(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	if err := p.CompareAndSwap("state", nil, []byte("one")); err != nil {
		t.Fatal(err)
	}
	for _, old := range [][]byte{nil, []byte("wrong"), {}} {
		if err := p.CompareAndSwap("state", old, []byte("bad")); !errors.Is(err, ErrConflict) {
			t.Fatal(err)
		}
	}
	if err := p.CompareAndSwap("state", []byte("one"), []byte{}); err != nil {
		t.Fatal(err)
	}
	if err := p.CompareAndSwap("state", nil, []byte("bad")); !errors.Is(err, ErrConflict) {
		t.Fatal("empty file mistaken for absence", err)
	}
	if err := p.CompareAndSwap("state", []byte{}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ReadFile("state"); !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	if err := p.CompareAndSwap("state", []byte{}, []byte("bad")); !errors.Is(err, ErrConflict) {
		t.Fatal(err)
	}
	if err := p.CompareAndSwap("state", nil, nil); err == nil {
		t.Fatal("empty transition")
	}
}

func TestReadFileLimit(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	value := bytes.Repeat([]byte("x"), 32)
	if err := p.WriteFile("state", value); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ReadFileLimit("state", 31); err == nil {
		t.Fatal("limit ignored")
	}
	got, err := p.ReadFileLimit("state", 32)
	if err != nil || !bytes.Equal(value, got) {
		t.Fatal(err)
	}
	for _, limit := range []int64{-1, 0, 1<<63 - 1} {
		if _, err := p.ReadFileLimit("state", limit); err == nil {
			t.Fatal("invalid limit")
		}
	}
	if _, err := p.ReadFileLimit("../state", 32); !errors.Is(err, ErrUnsafe) {
		t.Fatal("traversal", err)
	}
}

func TestCASRejectsUnsafeTargets(t *testing.T) {
	for _, kind := range []string{"symlink", "hardlink", "mode"} {
		t.Run(kind, func(t *testing.T) {
			p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
			if err := p.WriteFile("original", []byte("one")); err != nil {
				t.Fatal(err)
			}
			source, target := filepath.Join(p.Dir(), "original"), filepath.Join(p.Dir(), "target")
			var err error
			switch kind {
			case "symlink":
				err = os.Symlink(source, target)
			case "hardlink":
				err = os.Link(source, target)
			case "mode":
				err = os.WriteFile(target, []byte("one"), 0644)
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, next := range [][]byte{nil, []byte("bad")} {
				if err := p.CompareAndSwap("target", []byte("one"), next); !errors.Is(err, ErrUnsafe) {
					t.Fatal(err)
				}
			}
			if _, err := p.ReadFileLimit("target", 100); !errors.Is(err, ErrUnsafe) {
				t.Fatal(err)
			}
			b, err := os.ReadFile(source)
			if err != nil || string(b) != "one" {
				t.Fatal("modified source", err)
			}
		})
	}
}

func TestCASProcessHelper(t *testing.T) {
	dir := os.Getenv("MIXLAB_TEST_CAS_DIR")
	if dir == "" {
		return
	}
	p, err := Resolve(Options{ExactDir: dir}, Context{Kind: Principal})
	if err != nil {
		os.Exit(1)
	}
	err = p.CompareAndSwap("state", []byte("old"), []byte("new"))
	if errors.Is(err, ErrConflict) {
		os.Exit(3)
	}
	if err != nil {
		os.Exit(1)
	}
	os.Exit(0)
}

func TestCASAcrossProcesses(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	if err := p.WriteFile("state", []byte("old")); err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	var commands []*exec.Cmd
	for i := 0; i < 6; i++ {
		cmd := exec.Command(exe, "-test.run=^TestCASProcessHelper$", "-test.timeout=10s")
		cmd.Env = append(os.Environ(), "MIXLAB_TEST_CAS_DIR="+p.Dir())
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if cmd.ProcessState == nil {
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
			}
		})
		commands = append(commands, cmd)
	}
	wins := 0
	for _, cmd := range commands {
		err := cmd.Wait()
		if err == nil {
			wins++
			continue
		}
		var exit *exec.ExitError
		if !errors.As(err, &exit) || exit.ExitCode() != 3 {
			t.Errorf("child: %v", err)
		}
	}
	if wins != 1 {
		t.Fatalf("%d processes replaced one generation", wins)
	}
}
