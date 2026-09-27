package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

// withController is the composition owner. Only the authority application
// opens CA state; callbacks receive the controller and a pinned endpoint.
func withController(parent context.Context, home, exact, ca, endpoint string, stderr io.Writer, action func(context.Context, *principal.Store, string) error) (result error) {
	o, err := stateOptions(home, exact)
	if err != nil {
		return err
	}
	path, err := statehome.Discover(o, statehome.Context{Kind: statehome.Principal, Role: string(trust.Controller)})
	if err != nil {
		return err
	}
	p, err := principal.Open(path, time.Now())
	if err != nil {
		return err
	}
	defer func() { result = errors.Join(result, p.Close()) }()
	s, err := p.View(time.Now())
	if err != nil {
		return err
	}
	if s.Role != trust.Controller {
		return fmt.Errorf("controller principal required")
	}
	urls := s.Snapshot.Payload.Endpoints.Payload.URLs
	if endpoint == "" {
		if len(urls) != 1 {
			return fmt.Errorf("multiple signed authority endpoints; choose -authority-endpoint")
		}
		endpoint = urls[0]
	}
	if !slices.Contains(urls, endpoint) {
		return fmt.Errorf("authority endpoint is not root-signed")
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	if ca != "" {
		o, err := stateOptions(home, ca)
		if err != nil {
			return err
		}
		path, err := statehome.Discover(o, statehome.Context{Kind: statehome.Authority, ClusterID: s.Cluster})
		if err != nil {
			return err
		}
		a, err := clusterapp.OpenAuthority(ctx, path, time.Now())
		if err != nil {
			return err
		}
		defer func() { result = errors.Join(result, a.Close()) }()
		if a.Anchor.Cluster() != s.Cluster || a.Anchor.Fingerprint() != s.Fingerprint {
			return fmt.Errorf("local authority differs from controller pin")
		}
		u, err := url.Parse(endpoint)
		if err != nil {
			return err
		}
		address := u.Host
		if u.Port() == "" {
			address = net.JoinHostPort(u.Hostname(), "443")
		}
		l, err := net.Listen("tcp", address)
		if err != nil {
			return fmt.Errorf("temporary authority listener: %w; omit -cluster-state-dir when authority serve already runs", err)
		}
		done := make(chan error, 1)
		go func() { done <- clusterapp.ServeAuthority(ctx, l, a, time.Now); cancel() }()
		defer func() { cancel(); result = errors.Join(result, <-done) }()
	}
	if err := clusterapp.RefreshRemoteTrust(ctx, p, endpoint, time.Now); err != nil {
		// A current cached snapshot remains valid during an authority outage.
		// This matters especially for cancellation of an already running job.
		_, _ = fmt.Fprintln(stderr, "controller trust refresh:", err)
	}
	if _, _, err := p.Active(time.Now()); err != nil {
		return err
	}
	maintenance := make(chan error, 1)
	go func() {
		maintenance <- clusterapp.MaintainControllerTrust(ctx, path, p, time.Now, func(err error) { _, _ = fmt.Fprintln(stderr, err) })
		cancel()
	}()
	defer func() { cancel(); result = errors.Join(result, <-maintenance) }()
	return action(ctx, p, endpoint)
}
