//go:build darwin || linux

package local

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

func binding() wc.Binding {
	return wc.Binding{JobID: "job", AttemptID: "attempt", BinaryID: "mixlab", BuildID: "build", AssignmentDigest: [32]byte{1}}
}

func limits() Limits { return Limits{FrameBytes: 4096, Bytes: 65536, Messages: 16} }

func message(seq uint64) wc.Envelope {
	return wc.Envelope{Version: wc.Version, JobID: "job", AttemptID: "attempt", Sequence: seq,
		CorrelationID: "request", Kind: wc.KindProgress, PayloadKind: "test_progress", PayloadVersion: 1, Payload: []byte(`{"step":1}`)}
}

func bounded(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func directory(t *testing.T) statehome.Path {
	t.Helper()
	base, err := filepath.EvalSymlinks(os.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	// Darwin limits sockaddr_un paths to 104 bytes, unlike t.TempDir's names.
	if len(base) > 50 {
		base = "/private/tmp"
	}
	dir, err := os.MkdirTemp(base, "wc-ipc-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestLocalChild is executed in a distinct process. Only the non-secret socket
// path and test scenario are in the environment; the capability arrives on FD3.
func TestLocalChild(t *testing.T) {
	mode := os.Getenv("MIXLAB_LOCAL_TEST_CHILD")
	if mode == "" {
		return
	}
	f := os.NewFile(3, "worker-credential")
	if mode == "exit" {
		_ = f.Close()
		return
	}
	if mode == "stall" {
		time.Sleep(3 * time.Second)
		_ = f.Close()
		return
	}
	path := os.Getenv("MIXLAB_LOCAL_TEST_SOCKET")
	ctx := bounded(t)
	if mode == "proof" || mode == "binding" {
		defer func() { _ = f.Close() }()
		var proof [wc.CapabilityBytes]byte
		defer clear(proof[:])
		m, err := readHello(f, &proof)
		if err != nil {
			t.Fatal(err)
		}
		if mode == "proof" {
			proof[0] ^= 1
		} else {
			m.Binding.BuildID = "other-build"
		}
		conn, err := net.DialTimeout("unix", path, time.Second)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = conn.Close() }()
		_ = conn.SetDeadline(time.Now().Add(time.Second))
		if err := writeHello(conn, m, proof[:]); err != nil {
			t.Fatal(err)
		}
		var b [1]byte
		if _, err := conn.Read(b[:]); err == nil {
			t.Fatal("bad proof accepted")
		}
		return
	}
	c, err := Connect(ctx, path, f, limits())
	if mode == "peer" {
		if err == nil {
			_ = c.Close()
			t.Fatal("wrong child PID admitted")
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	if c.Binding() != binding() {
		t.Fatal("wrong assignment")
	}
	e, err := c.Receive(ctx)
	if err != nil || e.Sequence != 1 {
		t.Fatalf("assignment: %+v %v", e, err)
	}
	if mode == "replay" {
		if err := wc.WriteFrame(c.conn, message(1), 4096); err != nil {
			t.Fatal(err)
		}
		if err := wc.WriteFrame(c.conn, message(1), 4096); err != nil {
			t.Fatal(err)
		}
		return
	}
	if err := c.Send(ctx, message(1)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Receive(ctx); !errors.Is(err, io.EOF) {
		t.Fatalf("expected owner shutdown, got %v", err)
	}
}

func child(t *testing.T, s *Server, f *os.File, mode string) (*exec.Cmd, *bytes.Buffer) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestLocalChild$")
	cmd.Env = append(os.Environ(), "MIXLAB_LOCAL_TEST_CHILD="+mode, "MIXLAB_LOCAL_TEST_SOCKET="+s.Path())
	cmd.ExtraFiles = []*os.File{f}
	var log bytes.Buffer
	cmd.Stdout, cmd.Stderr = &log, &log
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if cmd.ProcessState == nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	})
	return cmd, &log
}

func TestNativeChildSession(t *testing.T) {
	for _, mode := range []string{"good", "proof", "binding", "peer", "replay"} {
		t.Run(mode, func(t *testing.T) {
			s, f, err := Listen(directory(t), limits())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = s.Close() })
			cmd, log := child(t, s, f, mode)
			peer := wc.PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Geteuid())}
			if mode == "peer" {
				peer.PID = os.Getpid()
			}
			c, err := s.Accept(bounded(t), binding(), peer)
			want := map[string]error{"proof": wc.ErrProof, "binding": wc.ErrBinding, "peer": wc.ErrPeer}[mode]
			if want != nil {
				if !errors.Is(err, want) {
					t.Fatalf("got %v want %v", err, want)
				}
			} else {
				if err != nil {
					t.Fatalf("admit: %v", err)
				}
				if err := c.Send(bounded(t), message(1)); err != nil {
					t.Fatal(err)
				}
				if _, err := c.Receive(bounded(t)); err != nil {
					t.Fatal(err)
				}
				if mode == "replay" {
					if _, err := c.Receive(bounded(t)); !errors.Is(err, wc.ErrSequence) {
						t.Fatalf("replay: %v", err)
					}
				}
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if err := cmd.Wait(); err != nil {
				t.Fatalf("child %v: %s", err, log.String())
			}
			if _, err := s.Accept(bounded(t), binding(), peer); !errors.Is(err, wc.ErrClosed) {
				t.Fatalf("reuse: %v", err)
			}
			if _, err := os.Lstat(s.Path()); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("socket leaked: %v", err)
			}
		})
	}
}

func TestStartupDeadlineAndChildExit(t *testing.T) {
	for _, mode := range []string{"stall", "exit"} {
		t.Run(mode, func(t *testing.T) {
			s, f, err := Listen(directory(t), limits())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			cmd, _ := child(t, s, f, mode)
			ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
			defer cancel()
			start := time.Now()
			_, err = s.Accept(ctx, binding(), wc.PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Geteuid())})
			if err == nil || time.Since(start) > 2*time.Second {
				t.Fatalf("unbounded startup: %v", err)
			}
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		})
	}
}

func TestListenerSafety(t *testing.T) {
	dir := directory(t)
	if err := os.Chmod(dir.Dir(), 0755); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Listen(dir, limits()); !errors.Is(err, statehome.ErrUnsafe) {
		t.Fatal(err)
	}
	if err := os.Chmod(dir.Dir(), 0700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir.Dir(), "control.sock")
	if err := os.WriteFile(path, []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Listen(dir, limits()); err == nil {
		t.Fatal("overwrote existing file")
	}
	value, _ := os.ReadFile(path)
	if string(value) != "keep" {
		t.Fatal("changed existing file")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	s, _, err := Listen(dir, limits())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("replacement"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	value, _ = os.ReadFile(path)
	if string(value) != "replacement" {
		t.Fatal("removed unrelated replacement")
	}
}

func TestConnectRejectsCounterfeitAgentAndBadBootstrap(t *testing.T) {
	for _, mode := range []string{"wrong-agent", "trailing-data", "partial", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			dir := directory(t)
			s, _, err := Listen(dir, limits())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			r, w, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = r.Close(); _ = w.Close() }()
			m := metadata{Binding: binding(), Agent: wc.PeerIdentity{PID: os.Getpid() + 100000, UID: uint32(os.Geteuid())}}
			if mode != "canceled" {
				if mode == "partial" {
					_, err = w.Write([]byte{0, 0})
				} else {
					err = writeHello(w, m, make([]byte, wc.CapabilityBytes))
				}
				if err != nil {
					t.Fatal(err)
				}
				if mode == "trailing-data" {
					if _, err := w.Write([]byte{1}); err != nil {
						t.Fatal(err)
					}
				}
				if err := w.Close(); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			_, err = Connect(ctx, s.Path(), r, limits())
			if err == nil {
				t.Fatal("invalid bootstrap accepted")
			}
			if mode == "wrong-agent" && !errors.Is(err, wc.ErrPeer) {
				t.Fatalf("wrong agent: %v", err)
			}
			if _, err := r.Stat(); !errors.Is(err, os.ErrClosed) {
				t.Fatal("credential descriptor left open")
			}
		})
	}
	f, err := os.CreateTemp(directory(t).Dir(), "not-a-pipe")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Connect(bounded(t), "/irrelevant", f, limits()); !errors.Is(err, ErrHandshake) {
		t.Fatal(err)
	}
}

func TestServerCloseUnblocksAccept(t *testing.T) {
	s, f, err := Listen(directory(t), limits())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	cmd, _ := child(t, s, f, "stall")
	result := make(chan error, 1)
	ctx := bounded(t)
	go func() {
		_, err := s.Accept(ctx, binding(), wc.PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Geteuid())})
		result <- err
	}()
	time.Sleep(20 * time.Millisecond)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-result:
		if err == nil {
			t.Fatal("admitted after close")
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not interrupt admission")
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
}
