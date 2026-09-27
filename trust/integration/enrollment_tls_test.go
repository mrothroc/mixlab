package integration

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

type handshakeResult struct {
	c   *tls.Conn
	err error
}

func tlsPair(t *testing.T, server, client *tls.Config) (*tls.Conn, *tls.Conn, error) {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	defer func() { _ = l.Close() }()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	result := make(chan handshakeResult, 1)
	go func() {
		raw, err := l.Accept()
		if err != nil {
			result <- handshakeResult{err: err}
			return
		}
		c := tls.Server(raw, server)
		err = c.HandshakeContext(ctx)
		result <- handshakeResult{c, err}
	}()
	raw, err := (&net.Dialer{}).DialContext(ctx, "tcp", l.Addr().String())
	check(t, err)
	cc := tls.Client(raw, client)
	clientErr := cc.HandshakeContext(ctx)
	if clientErr != nil {
		_ = cc.Close()
	}
	r := <-result
	t.Cleanup(func() {
		for _, c := range []*tls.Conn{cc, r.c} {
			if c != nil {
				_ = c.SetDeadline(time.Now())
				_ = c.Close()
			}
		}
	})
	return r.c, cc, errors.Join(clientErr, r.err)
}

func enrollmentConfigs(t *testing.T, f *tlsFixture) (*tls.Config, *tls.Config) {
	t.Helper()
	s, err := enrollmenttls.ServerConfig(f.server, f.clock)
	check(t, err)
	c, err := enrollmenttls.ClientConfig(func(chain [][]byte, now time.Time) error {
		_, err := f.verify(trust.Authority, f.serverID)(chain, now)
		return err
	}, f.clock)
	check(t, err)
	return s, c
}

func TestEnrollmentTLSExporterPairing(t *testing.T) {
	f := newTLSFixture(t)
	s, c := enrollmentConfigs(t, f)
	server, client, err := tlsPair(t, s, c)
	check(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	source, err := enrollmenttls.New(ctx, server)
	check(t, err)
	remote, err := enrollmenttls.New(ctx, client)
	check(t, err)
	t.Cleanup(func() { _ = source.Close(); _ = remote.Close() })
	pair := enrollment.PairingContext{Version: enrollment.PairingVersion, Fingerprint: f.a.Fingerprint(), Audience: "enrollment", Purpose: enrollment.NodeEnrollment, Role: trust.Node, RequestID: id(t), RequestHash: strings.Repeat("ab", 32), ClientNonce: bytes.Repeat([]byte{1}, 32), ServerNonce: bytes.Repeat([]byte{2}, 32)}
	digest, err := pair.Digest()
	check(t, err)
	se, err := source.Bind(digest)
	check(t, err)
	defer clear(se)
	ce, err := remote.Bind(digest)
	check(t, err)
	defer clear(ce)
	if !bytes.Equal(se, ce) {
		t.Fatal("TLS peers derived different exporters")
	}
	sp, sd, err := enrollment.PairingPresentation(pair, se)
	check(t, err)
	cp, cd, err := enrollment.PairingPresentation(pair, ce)
	check(t, err)
	if sp != cp || sd != cd || len(strings.Fields(sp)) != 5 {
		t.Fatal("SAS mismatch")
	}
	for _, ch := range []*enrollmenttls.Channel{source, remote} {
		n, err := ch.Network()
		check(t, err)
		if !n.Peer.IsLoopback() || !n.Local.IsLoopback() || n.Interface == "" {
			t.Fatal("missing observed network", n)
		}
		if _, err := ch.Bind(digest); err == nil {
			t.Fatal("exporter reused")
		}
	}
	changed := pair
	changed.Audience = "another-endpoint"
	_, other, err := enrollment.PairingPresentation(changed, se)
	check(t, err)
	if other == sd {
		t.Fatal("audience not bound")
	}
	server2, client2, err := tlsPair(t, s, c)
	check(t, err)
	source2, err := enrollmenttls.New(ctx, server2)
	check(t, err)
	t.Cleanup(func() { _ = source2.Close() })
	remote2, err := enrollmenttls.New(ctx, client2)
	check(t, err)
	t.Cleanup(func() { _ = remote2.Close() })
	se2, err := source2.Bind(digest)
	check(t, err)
	defer clear(se2)
	if bytes.Equal(se, se2) {
		t.Fatal("exporter moved to another connection")
	}
	check(t, remote2.Close())
	if _, err := remote2.Bind(digest); err == nil {
		t.Fatal("closed channel accepted")
	}
	cancel()
	if _, err := source.Network(); err == nil {
		t.Fatal("canceled channel accepted")
	}
}

func TestEnrollmentTLSProfileIsolation(t *testing.T) {
	f := newTLSFixture(t)
	s, c := enrollmentConfigs(t, f)
	if !s.SessionTicketsDisabled || !c.SessionTicketsDisabled || c.ClientSessionCache != nil {
		t.Fatal("resumption permitted")
	}
	if _, err := enrollmenttls.ClientConfig(nil, f.clock); err == nil {
		t.Fatal("source acceptance is optional")
	}
	_, normal := f.policies(t)
	if _, _, err := tlsPair(t, s, normal.ClientConfig()); err == nil {
		t.Fatal("normal control TLS accepted as enrollment")
	}
	client, err := enrollmenttls.ClientConfig(func([][]byte, time.Time) error { return errors.New("human rejected proposed root") }, f.clock)
	check(t, err)
	if _, _, err := tlsPair(t, s, client); err == nil {
		t.Fatal("rejected root accepted")
	}
	serverConn, clientConn, err := tlsPair(t, s, c)
	check(t, err)
	if _, err := enrollmenttls.New(context.Background(), serverConn); err == nil {
		t.Fatal("unbounded channel")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	cancel()
	if _, err := enrollmenttls.New(ctx, clientConn); err == nil {
		t.Fatal("expired channel")
	}
}
