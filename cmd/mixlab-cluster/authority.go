package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/signal"
	"slices"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

func runAuthorityServe(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	f := flag.NewFlagSet("mixlab-cluster authority serve", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "state root for unambiguous authority lookup")
	ca := f.String("cluster-state-dir", "", "exact cluster authority directory")
	listen := f.String("trust-listen", "", "signed authority host:port; defaults to sole signed endpoint")
	advertise := f.String("trust-advertise", "off", "mdns or off; clients still require root-signed authority endpoints")
	f.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster authority serve [options]")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster authority serve:", err); return 2 }
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster authority serve:", err); return 1 }
	if f.NArg() != 0 || (*advertise != "off" && *advertise != "mdns") {
		return invalid(fmt.Errorf("no positional arguments; advertisement must be mdns or off"))
	}
	opts, err := stateOptions(*home, *ca)
	if err != nil {
		return invalid(err)
	}
	p, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Authority})
	if err != nil {
		return invalid(err)
	}
	a, err := clusterapp.OpenAuthority(ctx, p, time.Now())
	if err != nil {
		return failed(err)
	}
	defer func() { _ = a.Close() }()
	v, err := a.Current(ctx, time.Now())
	if err != nil {
		return failed(err)
	}
	b, err := v.Bytes()
	if err != nil {
		return failed(err)
	}
	var snapshot trust.SignedSnapshot
	if err := json.Unmarshal(b, &snapshot); err != nil {
		return failed(err)
	}
	urls := snapshot.Payload.Endpoints.Payload.URLs
	endpoint := ""
	if *listen == "" {
		if len(urls) != 1 {
			return invalid(fmt.Errorf("multiple authority endpoints; specify -trust-listen"))
		}
		endpoint = urls[0]
		u, parseErr := url.Parse(endpoint)
		if parseErr != nil {
			return invalid(parseErr)
		}
		*listen = u.Host
		if u.Port() == "" {
			*listen = net.JoinHostPort(u.Hostname(), "443")
		}
	} else {
		endpoint, err = bootstrapEndpoint(*listen)
		if err != nil {
			return invalid(err)
		}
	}
	if !slices.Contains(urls, endpoint) {
		return invalid(fmt.Errorf("listener must match a root-signed authority endpoint"))
	}
	l, err := net.Listen("tcp", *listen)
	if err != nil {
		return failed(err)
	}
	defer func() { _ = l.Close() }()
	ad, err := advertiseListener(ctx, *advertise, l, discovery.Authority, a.Anchor.Cluster(), discovery.Claims{Cluster: a.Anchor.Cluster(), RootFingerprint: a.Anchor.Fingerprint()})
	if err != nil {
		return failed(err)
	}
	defer func() { _ = ad.Close() }()
	if err := json.NewEncoder(stdout).Encode(struct {
		Endpoint    string `json:"endpoint"`
		Fingerprint string `json:"fingerprint"`
	}{endpoint, a.Anchor.Fingerprint()}); err != nil {
		return failed(err)
	}
	if err := clusterapp.ServeAuthorityWithEvents(ctx, l, a, time.Now, func(err error) { _, _ = fmt.Fprintln(stderr, "authority renewal failed; retry scheduled:", err) }); err != nil {
		return failed(err)
	}
	return 0
}

func runRevoke(args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster revoke", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "state root for unambiguous authority lookup")
	ca := f.String("cluster-state-dir", "", "exact cluster authority directory")
	principalID := f.String("principal-id", "", "principal to revoke; excludes -certificate-serial")
	serial := f.String("certificate-serial", "", "leaf certificate serial to revoke")
	reason := f.String("revocation-reason", "", "required audit reason, printable ASCII")
	mode := f.String("revocation-mode", "prospective", "prospective or compromise; compromise invalidates historical proofs")
	pool := f.String("pool", "local", "local node pool only")
	discover := f.String("discover", "mdns", "mdns or off for best-effort signed snapshot push")
	var daemons endpointFlags
	f.Var(&daemons, "daemon-address", "explicit node host:port for signed snapshot push (repeatable)")
	f.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster revoke -principal-id ID|-certificate-serial SERIAL -revocation-reason TEXT [options]\nPublishes durable revocation, then pushes signed trust to reachable agents. Push failure never rolls back revocation.")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster revoke:", err); return 2 }
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster revoke:", err); return 1 }
	if f.NArg() != 0 || *pool != "local" || (*discover != "mdns" && *discover != "off") || (*principalID == "") == (*serial == "") || *reason == "" || (*mode != "prospective" && *mode != "compromise") {
		return invalid(fmt.Errorf("exactly one principal/serial, reason, and valid revocation mode required"))
	}
	opts, err := stateOptions(*home, *ca)
	if err != nil {
		return invalid(err)
	}
	p, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Authority})
	if err != nil {
		return invalid(err)
	}
	ctx := context.Background()
	a, err := clusterapp.OpenAuthority(ctx, p, time.Now())
	if err != nil {
		return failed(err)
	}
	defer func() { _ = a.Close() }()
	kind, id := "principal", *principalID
	if id == "" {
		kind, id = "certificate", *serial
	}
	s, err := a.Snapshots.Revoke(ctx, kind, id, *mode, *reason, time.Now())
	if err != nil {
		return failed(err)
	}
	pushes, pushErr := pushRevocation(ctx, a.Anchor, s, *discover, daemons)
	if err := json.NewEncoder(stdout).Encode(struct {
		Generation uint64               `json:"generation"`
		Kind       string               `json:"kind"`
		ID         string               `json:"id"`
		Mode       string               `json:"mode"`
		Pushed     bool                 `json:"pushed"`
		Delivery   []revocationDelivery `json:"delivery"`
	}{s.Payload.Generation, kind, id, *mode, len(pushes) > 0 && pushErr == nil, pushes}); err != nil {
		return failed(err)
	}
	if pushErr != nil {
		return failed(fmt.Errorf("revocation committed at generation %d; delivery incomplete: %w", s.Payload.Generation, pushErr))
	}
	return 0
}
