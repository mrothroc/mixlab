package integration

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/grouptransport"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/transport/ringproxy"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
)

func listenRingTest(t *testing.T) net.Listener {
	t.Helper()
	l, e := net.Listen("tcp4", "127.0.0.1:0")
	check(t, e)
	t.Cleanup(func() { _ = l.Close() })
	return l
}
func echoRingTest(t *testing.T, l net.Listener, counter *atomic.Int32) {
	t.Helper()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			counter.Add(1)
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer func() { _ = c.Close() }()
				_ = c.SetDeadline(time.Now().Add(5 * time.Second))
				_, _ = io.Copy(c, c)
			}()
		}
	}()
	t.Cleanup(func() { _ = l.Close(); wg.Wait() })
}

func TestRingRelayEncryptedStreamsAndRevocation(t *testing.T) {
	f := newTLSFixture(t)
	node := id(t)
	job := signedJobFixture(t, f, node, nodeagent.Lease{ID: id(t), Run: id(t)}).Manifest
	signed, identities := transportFixture(t, f, job)
	public := listenRingTest(t)
	remote := listenRingTest(t)
	local := listenRingTest(t)
	shim := listenRingTest(t)
	signed.Plan.Members[0].Endpoint = public.Addr().String()
	signed.Plan.Members[1].Endpoint = remote.Addr().String()
	signed.Plan.LoopbackPorts = []int{local.Addr().(*net.TCPAddr).Port, shim.Addr().(*net.TCPAddr).Port}
	check(t, shim.Close())
	q, err := signed.Plan.SigningRequest()
	check(t, err)
	signed.Proof, err = trust.SignPrincipalProof(f.a, f.client.Chain, f.client.Key, f.view, q, f.now)
	check(t, err)
	actor, err := trust.AuthenticatePrincipal(f.a, f.view, f.client.Chain, f.now)
	check(t, err)
	accepted, err := grouptransport.Accept(f.a, f.view, actor, job, identities[0].Chain, signed, f.now)
	check(t, err)
	current := func(context.Context, time.Time) (trust.Anchor, trust.VerifiedSnapshot, error) {
		f.mu.RLock()
		defer f.mu.RUnlock()
		return f.a, f.view, nil
	}
	r, err := ringproxy.New(ringproxy.Options{Accepted: accepted, LocalRank: 0, Identity: identities[0], CurrentTrust: current, Clock: f.clock, Public: public, ConnectTimeout: time.Second, MaxConnections: 8})
	check(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	finished := make(chan error, 1)
	go func() { finished <- r.Run(ctx) }()
	defer func() {
		cancel()
		_ = r.Close()
		select {
		case <-finished:
		case <-time.After(5 * time.Second):
			t.Error("relay leaked goroutines")
		}
	}()
	peerPolicy, err := ringtls.New(identities[1], func(chain [][]byte, now time.Time) error {
		a, v, e := current(context.Background(), now)
		if e != nil {
			return e
		}
		if !bytes.Equal(chain[0], identities[0].Chain[0]) {
			return errors.New("wrong leaf")
		}
		return trust.VerifyWorkload(a, v, chain, signed.Plan.Members[0].Binding, now)
	}, f.clock)
	check(t, err)
	var localConnections, remoteConnections atomic.Int32
	echoRingTest(t, local, &localConnections)
	echoRingTest(t, tls.NewListener(remote, peerPolicy.ServerConfig()), &remoteConnections)
	for _, direction := range []string{"outgoing", "incoming"} {
		t.Run(direction, func(t *testing.T) {
			var c net.Conn
			var err error
			if direction == "outgoing" {
				c, err = net.DialTimeout("tcp4", r.Addresses()[1][0], time.Second)
			} else {
				c, err = tls.DialWithDialer(&net.Dialer{Timeout: time.Second}, "tcp", public.Addr().String(), peerPolicy.ClientConfig())
			}
			check(t, err)
			defer func() { _ = c.Close() }()
			check(t, c.SetDeadline(time.Now().Add(3*time.Second)))
			payload := bytes.Repeat([]byte("opaque-collective-bytes"), 8192)
			writeDone := make(chan error, 1)
			go func() {
				_, e := c.Write(payload)
				if e == nil {
					e = c.(interface{ CloseWrite() error }).CloseWrite()
				}
				writeDone <- e
			}()
			got, err := io.ReadAll(c)
			check(t, err)
			check(t, <-writeDone)
			if !bytes.Equal(got, payload) {
				t.Fatal("relay changed bytes")
			}
		})
	}
	if localConnections.Load() != 1 || remoteConnections.Load() != 1 {
		t.Fatal("unexpected forwarding", localConnections.Load(), remoteConnections.Load())
	}
	// A current cluster controller is still not a workload member. Its
	// handshake must not open a plaintext connection into the local runtime.
	wrong, err := ringtls.New(ringtls.Identity{Chain: f.client.Chain, Key: f.client.Key}, func([][]byte, time.Time) error { return nil }, f.clock)
	check(t, err)
	bad, err := tls.DialWithDialer(&net.Dialer{Timeout: time.Second}, "tcp", public.Addr().String(), wrong.ClientConfig())
	if err == nil {
		_ = bad.SetDeadline(time.Now().Add(time.Second))
		_, _ = bad.Write([]byte("forbidden"))
		var b [1]byte
		_, err = bad.Read(b[:])
		_ = bad.Close()
	}
	if err == nil {
		t.Fatal("wrong workload authenticated")
	}
	if localConnections.Load() != 1 {
		t.Fatal("unauthenticated bytes reached MLX")
	}
	// Keep a stream open across revocation and require the relay to close it.
	c, err := net.DialTimeout("tcp4", r.Addresses()[1][0], time.Second)
	check(t, err)
	defer func() { _ = c.Close() }()
	check(t, c.SetDeadline(time.Now().Add(4*time.Second)))
	_, err = c.Write([]byte("live"))
	check(t, err)
	b := make([]byte, 4)
	_, err = io.ReadFull(c, b)
	check(t, err)
	f.mu.Lock()
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: signed.Plan.Members[1].Binding.Principal, Mode: "compromise", FirstGeneration: f.snapshot.Payload.Generation, Reason: "test"}}
	f.signSnapshot(t)
	f.mu.Unlock()
	select {
	case err := <-finished:
		if err == nil {
			t.Fatal("revocation not a transport failure")
		}
		finished <- err
	case <-time.After(3 * time.Second):
		t.Fatal("revoked live stream survived")
	}
	var last [1]byte
	if _, err := c.Read(last[:]); err == nil {
		t.Fatal("stream remained open")
	}
}
