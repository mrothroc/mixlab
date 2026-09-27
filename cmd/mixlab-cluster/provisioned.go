package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
)

func stateOptions(home, exact string) (statehome.Options, error) {
	wd, err := os.Getwd()
	if err != nil {
		return statehome.Options{}, err
	}
	u, err := os.UserHomeDir()
	if err != nil {
		return statehome.Options{}, err
	}
	return statehome.Options{Flag: home, ExactDir: exact, Env: os.Getenv("MIXLAB_STATE_HOME"), UserHome: u, WorkingDir: wd}, nil
}

func protectedFile(path string) (statehome.Path, string, error) {
	if path == "" {
		return statehome.Path{}, "", fmt.Errorf("protected file path required")
	}
	dir, name := filepath.Split(path)
	if name == "" || name == "." || name == ".." {
		return statehome.Path{}, "", fmt.Errorf("protected file name required")
	}
	if dir == "" {
		dir = "."
	}
	o, err := stateOptions("", dir)
	if err != nil {
		return statehome.Path{}, "", err
	}
	p, err := statehome.Resolve(o, statehome.Context{Kind: statehome.Enrollment})
	return p, name, err
}

func runInvite(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	return runInviteContext(ctx, args, stdout, stderr)
}

func runInviteContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster invite", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "state root for unambiguous authority lookup")
	ca := f.String("cluster-state-dir", "", "exact cluster authority directory")
	purpose := f.String("invite-purpose", "node-enrollment", "node-enrollment, controller-enrollment or coordinator-enrollment")
	output := f.String("invite-output", "", "new provisioning file in an owner-only directory")
	ttl := f.Duration("invite-ttl", 10*time.Minute, "one-use invitation lifetime, at most one hour")
	endpoint := f.String("bootstrap-endpoint", "", "exact signed authority endpoint; defaults to its sole URL")
	f.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster invite -invite-output PATH [options]\nServes one temporary TLS enrollment endpoint until consumed, expired or canceled.")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster invite:", err); return 2 }
	if f.NArg() != 0 || *output == "" || *ttl < time.Second || *ttl > time.Hour {
		return invalid(fmt.Errorf("output path and TTL in [1s,1h] required; no positional arguments"))
	}
	p := enrollment.Purpose(*purpose)
	if p != enrollment.NodeEnrollment && p != enrollment.ControllerEnrollment && p != enrollment.CoordinatorEnrollment {
		return invalid(fmt.Errorf("unsupported principal enrollment purpose"))
	}
	opts, err := stateOptions(*home, *ca)
	if err != nil {
		return invalid(err)
	}
	path, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Authority})
	if err != nil {
		return invalid(err)
	}
	dir, name, err := protectedFile(*output)
	if err != nil {
		return invalid(err)
	}
	if err := dir.Ensure(); err != nil {
		return invalid(err)
	}
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster invite:", err); return 1 }
	a, err := clusterapp.OpenAuthority(ctx, path, time.Now())
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
	if *endpoint == "" {
		if len(urls) != 1 {
			return invalid(fmt.Errorf("multiple authority endpoints; specify -bootstrap-endpoint"))
		}
		*endpoint = urls[0]
	}
	if !slices.Contains(urls, *endpoint) {
		return invalid(fmt.Errorf("bootstrap endpoint must be a signed authority endpoint"))
	}
	u, err := url.Parse(*endpoint)
	if err != nil || u.Scheme != "https" || u.Hostname() == "" {
		return invalid(fmt.Errorf("valid HTTPS authority endpoint required"))
	}
	address := u.Host
	if u.Port() == "" {
		address = net.JoinHostPort(u.Hostname(), "443")
	}
	l, err := net.Listen("tcp", address)
	if err != nil {
		return failed(err)
	}
	defer func() { _ = l.Close() }()
	i, err := a.Enrollment.Invite(ctx, p, *endpoint, snapshot.Payload.Endpoints.Payload.Audience, *ttl, v, time.Now())
	if err != nil {
		return failed(err)
	}
	defer i.Clear()
	if err := enrollment.WriteInvitation(dir, name, i, time.Now()); err != nil {
		return failed(err)
	}
	if err := json.NewEncoder(stdout).Encode(struct {
		ID, Endpoint, File string
		Expires            int64
	}{i.ID, i.Endpoint, filepath.Join(dir.Dir(), name), i.ExpiresAt}); err != nil {
		return failed(err)
	}
	if err := clusterapp.ServeInvitation(ctx, l, a, i, time.Now); err != nil {
		return failed(err)
	}
	return 0
}

func runEnroll(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	return runEnrollContext(ctx, args, stdout, stderr, terminalEnrollmentUI(stderr))
}

func runEnrollContext(ctx context.Context, args []string, stdout, stderr io.Writer, ui clusterapp.EnrollmentUI) int {
	return runEnrollWithDiscovery(ctx, args, stdout, stderr, ui, discovery.MDNS{})
}

func runEnrollWithDiscovery(ctx context.Context, args []string, stdout, stderr io.Writer, ui clusterapp.EnrollmentUI, provider discovery.Provider) int {
	f := flag.NewFlagSet("mixlab-cluster enroll", flag.ContinueOnError)
	f.SetOutput(stderr)
	finalDir := f.String("principal-state-dir", "", "absent destination for installed principal identity")
	policy := f.String("enrollment-policy", "", "explicit trusted-lan, provisioned or verified policy")
	file := f.String("enrollment-provisioning-file", "", "protected single-use provisioning file")
	endpoint := f.String("enrollment-coordinator", "", "explicit HTTPS enrollment coordinator URL")
	discover := f.String("enrollment-discover", "off", "mdns or off; mdns requires exactly one source and never supplies trust")
	purpose := f.String("enrollment-purpose", "node-enrollment", "node-enrollment, controller-enrollment or coordinator-enrollment")
	backend := f.String("key-backend", "", "new local key backend: keychain or file")
	f.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster enroll -enrollment-policy POLICY -principal-state-dir PATH [options]")
		f.PrintDefaults()
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster enroll:", err); return 2 }
	if f.NArg() != 0 || *finalDir == "" || (*discover != "off" && *discover != "mdns") {
		return invalid(fmt.Errorf("explicit policy, absent principal destination, and mdns/off discovery required"))
	}
	if *policy != "provisioned" && *policy != "trusted-lan" && *policy != "verified" {
		return invalid(fmt.Errorf("explicit enrollment policy required"))
	}
	if (*policy == "provisioned" && (*file == "" || *endpoint != "" || *discover != "off")) || (*policy != "provisioned" && (*file != "" || (*endpoint == "" && *discover == "off") || (*endpoint != "" && *discover == "mdns"))) {
		return invalid(fmt.Errorf("provisioned requires only a file; interactive policies require either a coordinator URL or mdns discovery"))
	}
	p := enrollment.Purpose(*purpose)
	if p != enrollment.NodeEnrollment && p != enrollment.ControllerEnrollment && p != enrollment.CoordinatorEnrollment {
		return invalid(fmt.Errorf("invalid principal purpose"))
	}
	if *policy == "trusted-lan" && p != enrollment.NodeEnrollment {
		return invalid(fmt.Errorf("trusted-lan is node-only"))
	}
	if *policy == "provisioned" {
		var set bool
		f.Visit(func(f *flag.Flag) {
			if f.Name == "enrollment-purpose" {
				set = true
			}
		})
		if set {
			return invalid(fmt.Errorf("provisioning file supplies the purpose"))
		}
	}
	opts, err := stateOptions("", *finalDir)
	if err != nil {
		return invalid(err)
	}
	final, err := statehome.Resolve(opts, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return invalid(err)
	}
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster enroll:", err); return 1 }
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return failed(err)
	}
	stage, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(final.Dir()), ".enrollment-"+hex.EncodeToString(nonce[:]))}, statehome.Context{Kind: statehome.Enrollment})
	if err != nil {
		return invalid(err)
	}
	var state principal.State
	if *policy == "provisioned" {
		dir, name, err := protectedFile(*file)
		if err != nil {
			return invalid(err)
		}
		i, err := enrollment.ReadInvitation(dir, name, time.Now())
		if err != nil {
			return failed(err)
		}
		defer i.Clear()
		state, err = clusterapp.EnrollProvisioned(ctx, i, stage, final, *backend, time.Now)
		if err != nil {
			return failed(err)
		}
		if err := enrollment.DestroyInvitation(dir, name, i); err != nil {
			return failed(fmt.Errorf("identity installed at %s but provisioning-file cleanup failed: %w", final.Dir(), err))
		}
	} else {
		if *discover == "mdns" {
			*endpoint, err = discoverEnrollmentEndpoint(ctx, provider)
			if err != nil {
				return failed(err)
			}
			_, _ = fmt.Fprintf(stderr, "Discovered enrollment endpoint: %s (untrusted address hint)\n", *endpoint)
		}
		selected := enrollment.Verified
		if *policy == "trusted-lan" {
			selected = enrollment.TrustedLAN
			_, _ = fmt.Fprintln(stderr, "WARNING: trusted-lan explicitly accepts the first valid coordinator root. Use only on an administrator-controlled isolated network.")
		}
		state, err = clusterapp.EnrollInteractive(ctx, clusterapp.InteractiveOptions{Endpoint: *endpoint, Policy: selected, Purpose: p, Backend: *backend, Stage: stage, Final: final, UI: ui}, time.Now)
		if err != nil {
			return failed(err)
		}
	}
	if err := json.NewEncoder(stdout).Encode(struct{ Cluster, Principal, Role, Directory string }{state.Cluster, state.Principal, string(state.Role), final.Dir()}); err != nil {
		return failed(err)
	}
	return 0
}
