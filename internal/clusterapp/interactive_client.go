package clusterapp

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/principal"
)

// EnrollmentUI collects local human decisions; it does not compute phrases,
// interpret policy, or approve on the source's behalf. Callbacks honor ctx.
type EnrollmentUI struct {
	AcceptRoot     func(context.Context, identity.Presentation) (bool, error)
	ConfirmRequest func(context.Context, enrollment.PendingApproval) (bool, error)
}

type InteractiveOptions struct {
	Endpoint     string
	Policy       enrollment.Policy
	Purpose      enrollment.Purpose
	Backend      string
	Stage, Final statehome.Path
	UI           EnrollmentUI
}

func EnrollInteractive(ctx context.Context, o InteractiveOptions, clock func() time.Time) (result principal.State, err error) {
	if clock == nil || (o.Policy != enrollment.Verified && o.Policy != enrollment.TrustedLAN) {
		return result, fmt.Errorf("explicit interactive policy and clock required")
	}
	if o.Policy == enrollment.Verified && (o.UI.AcceptRoot == nil || o.UI.ConfirmRequest == nil) {
		return result, fmt.Errorf("verified enrollment requires both local confirmations")
	}
	if o.Policy == enrollment.TrustedLAN && o.Purpose != enrollment.NodeEnrollment {
		return result, fmt.Errorf("trusted LAN is node-only")
	}
	ctx, cancel := context.WithTimeout(ctx, time.Hour)
	defer cancel()
	var a trust.Anchor
	conn, client, err := dialEnrollment(ctx, o.Endpoint, func(chain [][]byte, now time.Time) error {
		var e error
		a, e = trust.ProposedEnrollmentRoot(chain, trust.Coordinator, now)
		return e
	}, clock)
	if err != nil {
		return result, err
	}
	defer client.CloseIdleConnections()
	defer func() { _ = conn.SetDeadline(time.Now()); _ = conn.Close() }()
	stop := context.AfterFunc(ctx, func() { _ = conn.SetDeadline(time.Now()); _ = conn.Close() })
	defer stop()
	base, _ := url.Parse(o.Endpoint)
	route := func(path string) string { return base.ResolveReference(&url.URL{Path: path}).String() }
	var w enrollment.Window
	if err := enrollmentJSON(ctx, client, "GET", route("/v1/trust/enrollment-window"), nil, &w); err != nil {
		return result, err
	}
	if err := w.ValidateTarget(a, o.Endpoint, o.Policy, o.Purpose, clock()); err != nil {
		return result, err
	}
	life, done := context.WithTimeout(ctx, time.Unix(w.Expires, 0).Sub(clock()))
	defer done()
	if o.Policy == enrollment.Verified {
		p, err := identity.Cluster(a.Fingerprint())
		if err != nil {
			return result, err
		}
		accepted, err := o.UI.AcceptRoot(life, p)
		if err != nil {
			return result, err
		}
		if !accepted {
			return result, fmt.Errorf("cluster root comparison rejected")
		}
	}
	var snapshot trust.SignedSnapshot
	if err := enrollmentJSON(life, client, "GET", route("/v1/trust/snapshots/latest"), nil, &snapshot); err != nil {
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
	if peer.Role != trust.Coordinator {
		return result, fmt.Errorf("interactive source is not a coordinator")
	}
	channel, err := enrollmenttls.New(life, conn)
	if err != nil {
		return result, err
	}
	defer func() { _ = channel.Close() }()
	role := trust.Node
	switch o.Purpose {
	case enrollment.NodeEnrollment:
	case enrollment.ControllerEnrollment:
		role = trust.Controller
	case enrollment.CoordinatorEnrollment:
		role = trust.Coordinator
	default:
		return result, fmt.Errorf("unsupported principal purpose")
	}
	c, err := enrollee.Prepare(life, o.Stage, o.Final, a, role, o.Backend, clock())
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
	key, envelope, err := c.Keys()
	if err != nil {
		return result, err
	}
	q, err := enrollment.NewInteractiveRequest(a, w, o.Purpose, key, envelope, clock())
	if err != nil {
		return result, err
	}
	var p enrollment.Progress
	if err := enrollmentJSON(life, client, "POST", route("/v1/trust/enrollment-requests"), q, &p); err != nil {
		return result, err
	}
	phrase, digest, err := enrollment.ConfirmPairing(q, p, channel)
	if err != nil {
		return result, err
	}
	pairing := p.Pairing
	requestID := p.ID
	if o.Policy == enrollment.Verified {
		accepted, err := o.UI.ConfirmRequest(life, enrollment.PendingApproval{ID: requestID, RequestHash: pairing.RequestHash, Phrase: phrase, Digest: digest, Purpose: o.Purpose, Role: role})
		if err != nil {
			return result, err
		}
		if !accepted {
			return result, fmt.Errorf("request phrase comparison rejected")
		}
		body := struct {
			Digest string `json:"digest"`
		}{digest}
		if err := enrollmentJSON(life, client, "POST", route("/v1/trust/enrollment-requests/"+requestID+"/client-confirmations"), body, &p); err != nil {
			return result, err
		}
	}
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		if p.ID != requestID || p.PairingDigest != digest {
			return result, fmt.Errorf("enrollment response changed request")
		}
		switch p.Stage {
		case "issued":
			if p.Result == nil {
				return result, fmt.Errorf("issued enrollment missing credentials")
			}
			if err := c.CompleteInteractive(life, w, q, pairing, digest, *p.Result, clock()); err != nil {
				return result, err
			}
			installed, err := principal.Open(o.Final, clock())
			if err != nil {
				return result, err
			}
			defer func() { err = errors.Join(err, installed.Close()) }()
			return installed.View(clock())
		case "pending", "approved":
		default:
			return result, fmt.Errorf("enrollment request terminated: %s", p.Stage)
		}
		select {
		case <-life.Done():
			return result, life.Err()
		case <-ticker.C:
		}
		if err := enrollmentJSON(life, client, "GET", route("/v1/trust/enrollment-requests/"+requestID), nil, &p); err != nil {
			return result, err
		}
	}
}
