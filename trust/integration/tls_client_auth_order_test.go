package integration

import (
	"crypto/tls"
	"errors"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/transport/ringtls"
	"github.com/mrothroc/mixlab/trust"
)

func TestPinnedTLSRejectsServerBeforeClientIdentitySelection(t *testing.T) {
	for _, profile := range []string{"managed", "ring"} {
		t.Run(profile, func(t *testing.T) {
			f := newTLSFixture(t)
			var client *tls.Config
			if profile == "managed" {
				p, err := managedtls.New(f.client, func([][]byte, time.Time) (trust.AuthenticatedPrincipal, error) {
					return trust.AuthenticatedPrincipal{}, errors.New("untrusted server")
				}, f.clock)
				check(t, err)
				client = p.ClientConfig()
			} else {
				p, err := ringtls.New(f.client, func([][]byte, time.Time) error { return errors.New("untrusted server") }, f.clock)
				check(t, err)
				client = p.ClientConfig()
			}
			var identityRequests atomic.Int32
			client.GetClientCertificate = func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
				identityRequests.Add(1)
				return &client.Certificates[0], nil
			}
			listener := listenRingTest(t)
			server := &tls.Config{MinVersion: tls.VersionTLS13, MaxVersion: tls.VersionTLS13, ClientAuth: tls.RequireAnyClientCert, NextProtos: client.NextProtos, Time: f.clock, Certificates: []tls.Certificate{{Certificate: f.server.Chain, PrivateKey: f.server.Key}}}
			done := make(chan error, 1)
			go func() {
				raw, err := listener.Accept()
				if err != nil {
					done <- err
					return
				}
				defer func() { _ = raw.Close() }()
				_ = raw.SetDeadline(time.Now().Add(3 * time.Second))
				done <- tls.Server(raw, server).Handshake()
			}()
			conn, err := tls.DialWithDialer(&net.Dialer{Timeout: time.Second}, "tcp", listener.Addr().String(), client)
			if conn != nil {
				_ = conn.Close()
			}
			if err == nil {
				t.Error("untrusted server accepted")
			}
			if err := <-done; err == nil {
				t.Error("server completed rejected handshake")
			}
			if identityRequests.Load() != 0 {
				t.Fatal("client identity selected before server pin verification")
			}
		})
	}
}
