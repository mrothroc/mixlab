package clusterapp

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
)

func dialEnrollment(ctx context.Context, endpoint string, accept enrollmenttls.AcceptSource, clock func() time.Time) (*tls.Conn, *http.Client, error) {
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" {
		return nil, nil, fmt.Errorf("HTTPS enrollment endpoint required")
	}
	cfg, err := enrollmenttls.ClientConfig(accept, clock)
	if err != nil {
		return nil, nil, err
	}
	address := u.Host
	if u.Port() == "" {
		address = net.JoinHostPort(u.Hostname(), "443")
	}
	dial, done := context.WithTimeout(ctx, 10*time.Second)
	defer done()
	raw, err := (&net.Dialer{}).DialContext(dial, "tcp", address)
	if err != nil {
		return nil, nil, err
	}
	conn := tls.Client(raw, cfg)
	if err := conn.HandshakeContext(dial); err != nil {
		_ = raw.Close()
		return nil, nil, err
	}
	client, err := enrollmenttls.HTTPClient(conn, endpoint, 15*time.Second)
	if err != nil {
		_ = conn.Close()
		return nil, nil, err
	}
	return conn, client, nil
}

func enrollmentJSON(ctx context.Context, client *http.Client, method, target string, body, dst any) error {
	var b []byte
	var err error
	if body != nil {
		b, err = json.Marshal(body)
		if err != nil {
			return err
		}
		defer clear(b)
	}
	r, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(b))
	if err != nil {
		return err
	}
	r.Header.Set("Content-Type", "application/json")
	res, err := client.Do(r)
	if err != nil {
		return err
	}
	defer func() { _ = res.Body.Close() }()
	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("enrollment endpoint rejected request (HTTP %d)", res.StatusCode)
	}
	b, err = io.ReadAll(io.LimitReader(res.Body, 2*trust.MaxTrustBytes+1))
	if err != nil {
		return err
	}
	if len(b) > 2*trust.MaxTrustBytes {
		return fmt.Errorf("enrollment response too large")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("trailing enrollment response data")
	}
	return nil
}

// EnrollProvisioned sends the one-use secret only after the source is checked
// against the file's pin and a fresh root-verified snapshot on the same TLS
// connection. It does not delete the caller's provisioning file; the CLI does
// that exact protected-file cleanup after successful credential publication.
func EnrollProvisioned(ctx context.Context, i enrollment.Invitation, stage, final statehome.Path, backend string, clock func() time.Time) (result principal.State, err error) {
	if clock == nil {
		return result, fmt.Errorf("enrollment clock required")
	}
	a, err := i.ValidateTarget(i.Endpoint, i.Audience, clock())
	if err != nil {
		return result, err
	}
	base, err := url.Parse(i.Endpoint)
	if err != nil {
		return result, err
	}
	route := func(path string) string { return base.ResolveReference(&url.URL{Path: path}).String() }
	conn, client, err := dialEnrollment(ctx, i.Endpoint, func(chain [][]byte, now time.Time) error {
		return trust.VerifyProvisionalSource(a, chain, trust.Authority, now)
	}, clock)
	if err != nil {
		return result, err
	}
	defer client.CloseIdleConnections()
	defer func() { _ = conn.SetDeadline(time.Now()); _ = conn.Close() }()
	var snapshot trust.SignedSnapshot
	if err := enrollmentJSON(ctx, client, "GET", route("/v1/trust/snapshots/latest"), nil, &snapshot); err != nil {
		return result, err
	}
	v, err := trust.VerifySnapshot(a, snapshot, clock())
	if err != nil {
		return result, err
	}
	var chain [][]byte
	for _, c := range conn.ConnectionState().PeerCertificates {
		chain = append(chain, c.Raw)
	}
	peer, err := trust.AuthenticatePrincipal(a, v, chain, clock())
	if err != nil {
		return result, err
	}
	if peer.Role != trust.Authority || snapshot.Payload.Endpoints.Payload.Audience != i.Audience {
		return result, fmt.Errorf("provisioning authority audience/role mismatch")
	}
	listed := false
	for _, endpoint := range snapshot.Payload.Endpoints.Payload.URLs {
		if endpoint == i.Endpoint {
			listed = true
		}
	}
	if !listed {
		return result, fmt.Errorf("provisioning endpoint absent from signed trust")
	}
	c, err := enrollee.Prepare(ctx, stage, final, a, i.Role, backend, clock())
	if c != nil {
		defer func() {
			if err != nil {
				cleanup, done := context.WithTimeout(context.Background(), 5*time.Second)
				defer done()
				err = errors.Join(err, c.Abort(cleanup))
			}
			err = errors.Join(err, c.Close())
		}()
	}
	if err != nil {
		return result, err
	}
	k, envelope, err := c.Keys()
	if err != nil {
		return result, err
	}
	q, err := enrollment.NewRequest(i, k, envelope, clock())
	if err != nil {
		return result, err
	}
	var out enrollment.Result
	if err := enrollmentJSON(ctx, client, "POST", route("/v1/trust/enrollments/"+i.ID+"/consume"), consumeRequest{Request: q, Secret: i.Secret}, &out); err != nil {
		return result, fmt.Errorf("enrollment outcome uncertain; do not reuse the file blindly: %w", err)
	}
	if err := c.CompleteProvisioned(ctx, i, q, out, clock()); err != nil {
		return result, err
	}
	installed, err := principal.Open(final, clock())
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, installed.Close()) }()
	return installed.View(clock())
}
