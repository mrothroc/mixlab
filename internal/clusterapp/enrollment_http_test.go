package clusterapp

import (
	"bytes"
	"context"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestEnrollmentHTTPBoundFlow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	coordinator, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "coordinator")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	source, err := principal.Open(coordinator, testNow)
	check(t, err)
	defer func() { _ = source.Close() }()
	s, k, err := source.Active(testNow)
	check(t, err)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	endpoint := "https://" + l.Addr().String()
	v, err := a.Current(ctx, testNow)
	check(t, err)
	w, err := a.Enrollment.OpenWindow(ctx, enrollment.WindowOptions{Policy: enrollment.Verified, Endpoint: endpoint, Audience: "enrollment", MaxApprovals: 2}, v, testNow)
	check(t, err)
	done := make(chan error, 1)
	go func() {
		done <- ServeEnrollment(ctx, l, enrollmenttls.Identity{Chain: s.Chain, Key: k}, a.Enrollment, w, a.Current, func() time.Time { return testNow })
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			check(t, err)
		case <-time.After(10 * time.Second):
			t.Error("enrollment server did not clean up")
		}
	})
	connect := func() (*http.Client, *enrollmenttls.Channel) {
		raw, err := (&net.Dialer{}).DialContext(ctx, "tcp", l.Addr().String())
		check(t, err)
		cfg, err := enrollmenttls.ClientConfig(func(chain [][]byte, at time.Time) error {
			id, err := trust.AuthenticatePrincipal(a.Anchor, v, chain, at)
			if err == nil && (id.Role != trust.Coordinator || id.Principal != s.Principal) {
				return io.ErrUnexpectedEOF
			}
			return err
		}, func() time.Time { return testNow })
		check(t, err)
		conn := tls.Client(raw, cfg)
		check(t, conn.HandshakeContext(ctx))
		ch, err := enrollmenttls.New(ctx, conn)
		check(t, err)
		t.Cleanup(func() { _ = ch.Close() })
		client, err := enrollmenttls.HTTPClient(conn, endpoint, 3*time.Second)
		check(t, err)
		t.Cleanup(client.CloseIdleConnections)
		return client, ch
	}
	client, ch := connect()
	do := func(c *http.Client, method, path string, body any, dst any) int {
		t.Helper()
		var b []byte
		if body != nil {
			b, err = json.Marshal(body)
			check(t, err)
		}
		r, err := http.NewRequestWithContext(ctx, method, endpoint+path, bytes.NewReader(b))
		check(t, err)
		r.Header.Set("Content-Type", "application/json")
		res, err := c.Do(r)
		check(t, err)
		defer func() { _ = res.Body.Close() }()
		if res.StatusCode == 200 && dst != nil {
			check(t, json.NewDecoder(io.LimitReader(res.Body, 64<<10)).Decode(dst))
		} else {
			_, err = io.Copy(io.Discard, res.Body)
			check(t, err)
		}
		return res.StatusCode
	}
	var displayed enrollment.Window
	if got := do(client, "GET", "/v1/trust/enrollment-window", nil, &displayed); got != 200 || displayed.ID != w.ID {
		t.Fatal("window endpoint", got)
	}
	_, localKey, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	x, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	q, err := enrollment.NewInteractiveRequest(a.Anchor, w, enrollment.NodeEnrollment, localKey, x.PublicKey().Bytes(), testNow)
	check(t, err)
	var progress enrollment.Progress
	if got := do(client, "POST", "/v1/trust/enrollment-requests", q, &progress); got != 200 || progress.Stage != "pending" {
		t.Fatal("submit", got, progress.Stage)
	}
	digest, err := progress.Pairing.Digest()
	check(t, err)
	exporter, err := ch.Bind(digest)
	check(t, err)
	_, confirmation, err := enrollment.PairingPresentation(progress.Pairing, exporter)
	clear(exporter)
	check(t, err)
	other, _ := connect()
	if got := do(other, "GET", "/v1/trust/enrollment-requests/"+progress.ID, nil, nil); got != 400 {
		t.Fatal("request moved connections", got)
	}
	if got := do(client, "POST", "/v1/trust/enrollment-requests/"+progress.ID+"/approve", struct{}{}, nil); got != 404 {
		t.Fatal("remote admin route exists", got)
	}
	if got := do(client, "POST", "/v1/trust/enrollment-requests/"+progress.ID+"/client-confirmations", struct {
		Digest string `json:"digest"`
	}{confirmation}, &progress); got != 200 {
		t.Fatal("confirmation", got)
	}
	if progress.Result != nil {
		t.Fatal("remote client self-approved")
	}
	_, err = a.Enrollment.Decide(ctx, progress.ID, confirmation, true, v, testNow)
	check(t, err)
	if got := do(client, "GET", "/v1/trust/enrollment-requests/"+progress.ID, nil, &progress); got != 200 || progress.Result == nil {
		t.Fatal("poll", got)
	}
	_, err = enrollment.ValidateInteractiveResult(a.Anchor, w, q, progress.Pairing, confirmation, *progress.Result, testNow)
	check(t, err)
	bad, err := http.NewRequest("GET", "https://another.invalid/v1/trust/enrollment-window", nil)
	check(t, err)
	if _, err := client.Do(bad); err == nil {
		t.Fatal("client changed enrollment origin")
	}
	check(t, ch.Close())
	if _, err := client.Get(endpoint + "/v1/trust/enrollment-window"); err == nil {
		t.Fatal("client silently reconnected")
	}
}
