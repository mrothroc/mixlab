package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/principal"
)

type repeatedFlag []string

func (v *repeatedFlag) String() string     { return fmt.Sprint([]string(*v)) }
func (v *repeatedFlag) Set(s string) error { *v = append(*v, s); return nil }

func runEnrollmentServe(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	return runEnrollmentServeContext(ctx, args, stdout, stderr, func(ctx context.Context, p enrollment.PendingApproval) (bool, error) {
		return terminalConfirm(ctx, stderr, requestPresentation(p), p.ID+" "+p.Phrase)
	})
}

func runEnrollmentServeContext(ctx context.Context, args []string, stdout, stderr io.Writer, approve func(context.Context, enrollment.PendingApproval) (bool, error)) (code int) {
	f := flag.NewFlagSet("mixlab-cluster enrollment serve", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "state root for unambiguous authority lookup")
	ca := f.String("cluster-state-dir", "", "exact cluster authority directory")
	principalDir := f.String("principal-state-dir", "", "source coordinator identity directory")
	listen := f.String("enrollment-listen", "127.0.0.1:7444", "concrete reachable host:port for temporary enrollment")
	advertise := f.String("enrollment-advertise", "off", "mdns or off; discovery hints never select trust or policy")
	policy := f.String("enrollment-policy", "", "trusted-lan or verified (required)")
	ttl := f.Duration("enrollment-ttl", 10*time.Minute, "window lifetime, at most one hour")
	maxNodes := f.Int("enrollment-max-nodes", 1, "maximum approvals in this window")
	iface := f.String("enrollment-interface", "", "interface owning the listener IP, not a physical-ingress filter; required for trusted-lan")
	var cidrs, purposes repeatedFlag
	f.Var(&cidrs, "enrollment-cidr", "allowed private CIDR; repeatable, required for trusted-lan")
	f.Var(&purposes, "enrollment-allow-purpose", "allowed principal purpose; repeatable, default node-enrollment")
	f.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster enrollment serve -enrollment-policy POLICY -principal-state-dir PATH [options]")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster enrollment serve:", err); return 2 }
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster enrollment serve:", err); return 1 }
	if f.NArg() != 0 || *principalDir == "" || (*advertise != "off" && *advertise != "mdns") || (*policy != "trusted-lan" && *policy != "verified") || *ttl < time.Second || *ttl > time.Hour || *maxNodes < 1 || *maxNodes > 256 {
		return invalid(fmt.Errorf("explicit policy/coordinator, mdns/off advertisement, TTL [1s,1h], and approvals [1,256] required"))
	}
	selected := enrollment.Verified
	if *policy == "trusted-lan" {
		selected = enrollment.TrustedLAN
	}
	if selected == enrollment.TrustedLAN && (*iface == "" || len(cidrs) == 0) {
		return invalid(fmt.Errorf("trusted-lan requires interface and CIDRs"))
	}
	if selected == enrollment.TrustedLAN {
		host, _, err := net.SplitHostPort(*listen)
		ip := net.ParseIP(host)
		if err != nil || ip == nil || (!ip.IsPrivate() && !ip.IsLoopback() && !ip.IsLinkLocalUnicast()) {
			return invalid(fmt.Errorf("trusted-lan requires an explicit private, link-local or loopback listener IP"))
		}
	}
	endpoint, err := bootstrapEndpoint(*listen)
	if err != nil {
		return invalid(err)
	}
	opts, err := stateOptions(*home, *ca)
	if err != nil {
		return invalid(err)
	}
	path, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Authority})
	if err != nil {
		return invalid(err)
	}
	po, err := stateOptions("", *principalDir)
	if err != nil {
		return invalid(err)
	}
	pp, err := statehome.Resolve(po, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return invalid(err)
	}
	a, err := clusterapp.OpenAuthority(ctx, path, time.Now())
	if err != nil {
		return failed(err)
	}
	defer func() { _ = a.Close() }()
	source, err := principal.Open(pp, time.Now())
	if err != nil {
		return failed(err)
	}
	defer func() { _ = source.Close() }()
	local, err := a.EnrollmentSource(ctx, source, time.Now())
	if err != nil {
		return failed(err)
	}
	l, err := net.Listen("tcp", *listen)
	if err != nil {
		return failed(err)
	}
	defer func() { _ = l.Close() }()
	if *advertise == "mdns" && endpoint != "https://"+l.Addr().String() {
		return invalid(fmt.Errorf("mDNS enrollment requires the canonical literal listener IP and port"))
	}
	if err := enrollmenttls.ValidateListener(l, *iface); err != nil {
		return failed(err)
	}
	v, err := a.Current(ctx, time.Now())
	if err != nil {
		return failed(err)
	}
	ps := make([]enrollment.Purpose, len(purposes))
	for i, p := range purposes {
		ps[i] = enrollment.Purpose(p)
	}
	w, err := a.Enrollment.OpenWindow(ctx, enrollment.WindowOptions{Policy: selected, Endpoint: endpoint, Audience: "mixlab-enrollment", Purposes: ps, TTL: *ttl, MaxApprovals: *maxNodes, Interface: *iface, CIDRs: cidrs}, v, time.Now())
	if err != nil {
		return failed(err)
	}
	defer func() {
		cleanup, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		v, err := a.Current(cleanup, time.Now())
		if err == nil {
			err = a.Enrollment.CloseWindow(cleanup, w.ID, v, time.Now())
		}
		if err != nil {
			_, _ = fmt.Fprintln(stderr, "enrollment window cleanup requires attention:", err)
			if code == 0 {
				code = 1
			}
		}
	}()
	presentation, err := identity.Cluster(a.Anchor.Fingerprint())
	if err != nil {
		return failed(err)
	}
	if err := json.NewEncoder(stdout).Encode(struct {
		Window   enrollment.Window     `json:"window"`
		Identity identity.Presentation `json:"identity"`
	}{w, presentation}); err != nil {
		return failed(err)
	}
	if selected == enrollment.TrustedLAN {
		_, _ = fmt.Fprintln(stderr, "WARNING: trusted-lan automatically approves eligible nodes at the selected local address from allowed CIDRs. It does not verify physical ingress or prevent routed/forwarded access. Use only on a trusted administrator-controlled LAN.")
	}
	life, cancel := context.WithTimeout(ctx, *ttl)
	defer cancel()
	ad, err := advertiseListener(life, *advertise, l, discovery.Enrollment, w.ID, discovery.Claims{Cluster: w.Cluster, RootFingerprint: w.Fingerprint, Role: "coordinator", Audience: w.Audience, Policies: []string{string(w.Policy)}})
	if err != nil {
		return failed(err)
	}
	defer func() { _ = ad.Close() }()
	current := func(ctx context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
		if _, err := a.EnrollmentSource(ctx, source, now); err != nil {
			return trust.VerifiedSnapshot{}, err
		}
		return a.Current(ctx, now)
	}
	done := make(chan error, 1)
	go func() {
		done <- clusterapp.ServeEnrollment(life, l, local, a.Enrollment, w, current, time.Now)
		cancel()
	}()
	if selected == enrollment.Verified {
		err = approvePending(life, a, w, approve)
		cancel()
	}
	serveErr := <-done
	if err != nil {
		return failed(err)
	}
	if serveErr != nil {
		return failed(serveErr)
	}
	return 0
}

func approvePending(ctx context.Context, a *clusterapp.Authority, w enrollment.Window, approve func(context.Context, enrollment.PendingApproval) (bool, error)) (result error) {
	defer func() {
		if ctx.Err() != nil && errors.Is(result, ctx.Err()) {
			result = nil
		}
	}()
	if approve == nil {
		return fmt.Errorf("administrator approval UI required")
	}
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
		}
		v, err := a.Current(ctx, time.Now())
		if err != nil {
			return err
		}
		pending, err := a.Enrollment.Pending(ctx, w.ID, v, time.Now())
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		for _, p := range pending {
			accepted, err := approve(ctx, p)
			if err != nil {
				if ctx.Err() != nil {
					return nil
				}
				return err
			}
			v, err := a.Current(ctx, time.Now())
			if err != nil {
				return err
			}
			if _, err := a.Enrollment.Decide(ctx, p.ID, p.Digest, accepted, v, time.Now()); err != nil {
				return err
			}
		}
	}
}
