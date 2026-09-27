//go:build darwin || linux

package local

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
)

// Server owns a single attempt's listener, bootstrap pipe, and connection.
// Close it when the supervised child terminates, including failed startup.
type Server struct {
	mu              sync.Mutex
	listener        *net.UnixListener
	read, write     *os.File
	conn            *net.UnixConn
	path            string
	inode           os.FileInfo
	transient       string
	transientInode  os.FileInfo
	started, closed bool
	limits          Limits
}

// Listen creates control.sock inside an already-secure per-attempt directory.
// Long persistent paths use a fresh private temporary socket directory instead;
// authentication still binds the exact child and immutable attempt capability.
// It does not repair permissions, overwrite sockets, or create persistent state.
// The returned file belongs only in the child's ExtraFiles, never argv/env.
// Accept closes the parent's copy after the caller has started the child.
func Listen(dir statehome.Path, limits Limits) (*Server, *os.File, error) {
	if err := limits.validate(); err != nil {
		return nil, nil, err
	}
	if err := dir.Validate(); err != nil {
		return nil, nil, err
	}
	path := filepath.Join(dir.Dir(), "control.sock")
	if len(path) >= 100 {
		return listenShort(limits)
	}
	l, err := net.ListenUnix("unix", &net.UnixAddr{Name: path, Net: "unix"})
	if err != nil {
		return nil, nil, err
	}
	l.SetUnlinkOnClose(false)
	s := &Server{listener: l, path: path, limits: limits}
	s.inode, err = os.Lstat(path)
	if err == nil {
		err = os.Chmod(path, 0600)
	}
	if err == nil {
		err = dir.Validate()
	}
	if err == nil {
		s.read, s.write, err = os.Pipe()
	}
	if err != nil {
		_ = s.Close()
		return nil, nil, err
	}
	return s, s.read, nil
}

func (s *Server) Path() string { return s.path }

func (s *Server) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	var result error
	for _, closer := range []io.Closer{s.listener, s.read, s.write, s.conn} {
		// Explicit checks avoid typed-nil interfaces.
		switch value := closer.(type) {
		case *net.UnixListener:
			if value == nil {
				continue
			}
		case *net.UnixConn:
			if value == nil {
				continue
			}
		case *os.File:
			if value == nil {
				continue
			}
		}
		if err := closer.Close(); err != nil && !errors.Is(err, net.ErrClosed) && !errors.Is(err, os.ErrClosed) {
			result = errors.Join(result, err)
		}
	}
	// Never unlink somebody else's replacement at the same path.
	if current, err := os.Lstat(s.path); err == nil && s.inode != nil && os.SameFile(current, s.inode) {
		result = errors.Join(result, os.Remove(s.path))
	}
	if s.transient != "" {
		if current, err := os.Lstat(s.transient); err == nil && os.SameFile(current, s.transientInode) {
			result = errors.Join(result, os.Remove(s.transient))
		}
	}
	return result
}

// Accept must follow child Start so expected PID is actual kernel evidence.
// The caller supervises that child throughout admission: if it exits, close
// this Server and discard admission. Once admitted, the listener is closed;
// hosting may drain a bounded terminal outcome before closing the connection.
// Every attempt, even failed authentication, consumes this Server. There is no
// reconnect or numeric-PID-reuse fallback.
func (s *Server) Accept(ctx context.Context, binding wc.Binding, expected wc.PeerIdentity) (out *Conn, err error) {
	s.mu.Lock()
	if s.closed || s.started {
		s.mu.Unlock()
		return nil, wc.ErrClosed
	}
	s.started = true
	s.mu.Unlock()
	defer func() {
		if err != nil {
			_ = s.Close()
		}
	}()
	d, err := deadline(ctx)
	if err != nil {
		return nil, err
	}
	stop := context.AfterFunc(ctx, func() { _ = s.Close() })
	defer stop()
	session, proof, err := wc.NewSession(binding, expected)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	defer clear(proof[:])
	m := metadata{Binding: binding, Agent: wc.PeerIdentity{PID: os.Getpid(), UID: uint32(os.Geteuid())}}
	_ = s.read.Close()
	if err = s.write.SetWriteDeadline(d); err != nil {
		return nil, err
	}
	if err = writeHello(s.write, m, proof[:]); err != nil {
		return nil, err
	}
	clear(proof[:])
	if err = s.write.Close(); err != nil {
		return nil, err
	}
	if err = s.listener.SetDeadline(d); err != nil {
		return nil, err
	}
	c, err := s.listener.AcceptUnix()
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		_ = c.Close()
		return nil, wc.ErrClosed
	}
	s.conn = c
	s.mu.Unlock()
	_ = s.listener.Close()
	if err = c.SetDeadline(d); err != nil {
		return nil, err
	}
	peer, err := inspect(c)
	if err != nil {
		return nil, err
	}
	if err = wc.ValidatePeerIdentity(expected, peer); err != nil {
		return nil, err
	}
	claimed, err := readHello(c, &proof)
	if err != nil {
		return nil, err
	}
	if claimed.Agent != m.Agent {
		return nil, wc.ErrPeer
	}
	if err = session.Authenticate(claimed.Binding, peer, proof[:]); err != nil {
		return nil, err
	}
	clear(proof[:])
	if err = writeAll(c, []byte(ready)); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return newConn(c, binding, s.limits), nil
}

// Connect consumes an inherited anonymous-pipe read descriptor and returns only
// after mutual local admission. Non-secret socket path may be passed in argv.
// limits are worker-side resource policy, not a claim read from the wire.
func Connect(ctx context.Context, socket string, credential *os.File, limits Limits) (out *Conn, err error) {
	if credential == nil {
		return nil, ErrHandshake
	}
	defer func() { _ = credential.Close() }()
	if err = limits.validate(); err != nil {
		return nil, err
	}
	d, err := deadline(ctx)
	if err != nil {
		return nil, err
	}
	info, err := credential.Stat()
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeNamedPipe == 0 {
		return nil, ErrHandshake
	}
	pollable, err := pollablePipe(credential)
	if err != nil {
		return nil, err
	}
	_ = credential.Close()
	credential = pollable
	stopPipe := context.AfterFunc(ctx, func() { _ = credential.Close() })
	defer stopPipe()
	if err = credential.SetReadDeadline(d); err != nil {
		return nil, err
	}
	var proof [wc.CapabilityBytes]byte
	defer clear(proof[:])
	m, err := readHello(credential, &proof)
	if err != nil {
		return nil, err
	}
	var extra [1]byte
	if n, readErr := credential.Read(extra[:]); n != 0 || readErr != io.EOF {
		return nil, ErrHandshake
	}
	_ = credential.Close()
	dir, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Dir(socket)}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return nil, err
	}
	if err = dir.Validate(); err != nil {
		return nil, err
	}
	info, err = os.Lstat(socket)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSocket == 0 || info.Mode().Perm() != 0600 {
		return nil, statehome.ErrUnsafe
	}
	n, err := (&net.Dialer{}).DialContext(ctx, "unix", socket)
	if err != nil {
		return nil, err
	}
	c := n.(*net.UnixConn)
	defer func() {
		if err != nil {
			_ = c.Close()
		}
	}()
	stopConn := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stopConn()
	if err = c.SetDeadline(d); err != nil {
		return nil, err
	}
	peer, err := inspect(c)
	if err != nil {
		return nil, err
	}
	if err = wc.ValidatePeerIdentity(m.Agent, peer); err != nil {
		return nil, err
	}
	if err = writeHello(c, m, proof[:]); err != nil {
		return nil, err
	}
	clear(proof[:])
	ack := make([]byte, len(ready))
	if _, err = io.ReadFull(c, ack); err != nil {
		return nil, err
	}
	if !bytes.Equal(ack, []byte(ready)) {
		return nil, ErrHandshake
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return newConn(c, m.Binding, limits), nil
}

func inspect(c *net.UnixConn) (wc.PeerIdentity, error) {
	raw, err := c.SyscallConn()
	if err != nil {
		return wc.PeerIdentity{}, err
	}
	return wc.InspectPeerIdentity(raw)
}
