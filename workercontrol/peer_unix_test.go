//go:build darwin || linux

package workercontrol

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// A separate process proves that inspection does not accidentally return the
// inspecting process's own PID. No capability is sent via argv/environment.
func TestPeerInspectionChild(t *testing.T) {
	path := os.Getenv("WORKERCONTROL_TEST_PEER_SOCKET")
	if path == "" {
		return
	}
	conn, err := net.DialTimeout("unix", path, 5*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer closePeerTestResource(t, conn)
	if err := conn.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var done [1]byte
	if _, err := io.ReadFull(conn, done[:]); err != nil {
		t.Fatal(err)
	}
}

func TestNativePeerInspectionAndAdmission(t *testing.T) {
	// Keep paths below Darwin's sockaddr_un limit; MkdirTemp is owner-only.
	dir, err := os.MkdirTemp("/tmp", "wc-peer-")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("remove peer test directory: %v", err)
		}
	}()
	path := filepath.Join(dir, "socket")
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer closePeerTestResource(t, listener)
	if err := os.Chmod(path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := listener.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, "-test.run=^TestPeerInspectionChild$")
	cmd.Env = append(os.Environ(), "WORKERCONTROL_TEST_PEER_SOCKET="+path)
	var childOutput bytes.Buffer
	cmd.Stdout, cmd.Stderr = &childOutput, &childOutput
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		cancel()
		if !waited {
			_ = cmd.Wait()
		}
	}()
	conn, err := listener.AcceptUnix()
	if err != nil {
		t.Fatal(err)
	}
	defer closePeerTestResource(t, conn)
	raw, err := conn.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	peer, err := InspectPeerIdentity(raw)
	if err != nil {
		t.Fatal(err)
	}
	expected := PeerIdentity{PID: cmd.Process.Pid, UID: uint32(os.Geteuid())}
	if err := ValidatePeerIdentity(expected, peer); err != nil {
		t.Fatalf("got %+v want %+v: %v", peer, expected, err)
	}
	s, proof, err := NewSession(testBinding(), expected)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Authenticate(testBinding(), peer, proof[:]); err != nil {
		t.Fatal(err)
	}
	s.Close()
	// Valid proof cannot compensate for a different supervised child/UID.
	for _, wrong := range []PeerIdentity{{PID: os.Getpid(), UID: expected.UID}, {PID: expected.PID, UID: expected.UID ^ 1}} {
		s, proof, err := NewSession(testBinding(), wrong)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Authenticate(testBinding(), peer, proof[:]); !errors.Is(err, ErrPeer) {
			t.Fatal(err)
		}
	}
	if _, err := conn.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}
	if err := conn.Close(); err != nil {
		t.Fatal(err)
	}
	err = cmd.Wait()
	waited = true
	if err != nil {
		t.Fatalf("child: %v\n%s", err, childOutput.String())
	}
	if _, err := InspectPeerIdentity(raw); !errors.Is(err, ErrPeerInspection) {
		t.Fatal(err)
	}
}

func TestNativePeerInspectionRejectsNonConnectedStreams(t *testing.T) {
	file, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer closePeerTestResource(t, file)
	raw, err := file.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := InspectPeerIdentity(raw); !errors.Is(err, ErrPeerInspection) {
		t.Fatal(err)
	}
	for _, kind := range []int{syscall.SOCK_DGRAM, syscall.SOCK_STREAM} {
		fd, err := syscall.Socket(syscall.AF_UNIX, kind, 0)
		if err != nil {
			t.Fatal(err)
		}
		f := os.NewFile(uintptr(fd), "unconnected-socket")
		raw, err := f.SyscallConn()
		if err != nil {
			closePeerTestResource(t, f)
			t.Fatal(err)
		}
		_, err = InspectPeerIdentity(raw)
		closePeerTestResource(t, f)
		if !errors.Is(err, ErrPeerInspection) {
			t.Fatal(err)
		}
	}
	// No IP listener is opened; even an unconnected IP socket must fail closed.
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_STREAM, 0)
	if err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(fd), "ip-socket")
	defer closePeerTestResource(t, f)
	raw, err = f.SyscallConn()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := InspectPeerIdentity(raw); !errors.Is(err, ErrPeerInspection) {
		t.Fatal(err)
	}
}

func closePeerTestResource(t *testing.T, resource io.Closer) {
	t.Helper()
	if err := resource.Close(); err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, os.ErrClosed) {
		t.Errorf("close peer test resource: %v", err)
	}
}
