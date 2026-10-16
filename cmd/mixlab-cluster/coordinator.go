package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterservice"
	"github.com/mrothroc/mixlab/statehome"
)

func runCoordinator(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stderr, "coordinator requires init, serve, install, start, stop, status or uninstall")
		return 2
	}
	action := args[0]
	if action != "init" && action != "serve" && action != "install" {
		if isServiceAction(action) {
			return runService("coordinator", action, args[1:], stdout, stderr)
		}
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return runCoordinatorContext(ctx, action, args[1:], stdout, stderr)
}

func runCoordinatorContext(ctx context.Context, action string, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster coordinator "+action, flag.ContinueOnError)
	f.SetOutput(stderr)
	paths := map[string]*string{}
	for _, name := range clusterservice.CoordinatorPathFlags() {
		paths[name] = f.String(name, "", "explicit independent absolute context directory")
	}
	listen := f.String("coordinator-listen", "", "concrete reachable listener host:port; also the advertised HTTPS origin")
	var binary string
	if action == "install" {
		f.StringVar(&binary, "cluster-binary", "", "stable signed cluster executable (optional)")
	}
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if f.NArg() != 0 {
		return 2
	}
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "coordinator:", err); return 1 }
	argv := []string{"coordinator", "serve"}
	for _, name := range clusterservice.CoordinatorPathFlags() {
		argv = append(argv, "-"+name, *paths[name])
	}
	argv = append(argv, "-coordinator-listen", *listen)
	// The same closed allowlist validates foreground and installed execution.
	if err := (clusterservice.Spec{Role: "coordinator", Binary: "/mixlab-cluster", Arguments: argv}).Validate(); err != nil {
		return fail(err)
	}
	host, _, _ := net.SplitHostPort(*listen)
	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		return fail(fmt.Errorf("coordinator-listen requires a reachable local address, not a wildcard"))
	}
	resolved := map[string]statehome.Path{}
	for name, path := range paths {
		p, err := statehome.Resolve(statehome.Options{ExactDir: *path}, statehome.Context{Kind: statehome.Principal})
		if err != nil {
			return fail(err)
		}
		resolved[name] = p
	}
	o := clusterapp.DiLoCoCoordinatorOptions{Principal: resolved["principal-state-dir"], AdmissionPrincipal: resolved["admission-principal-state-dir"], State: resolved["coordinator-state-dir"], Admission: resolved["admission-state-dir"], Checkpoints: resolved["checkpoint-dir"], Artifacts: resolved["artifact-dir"], Endpoint: "https://" + *listen, Clock: time.Now, Event: func(err error) { _, _ = fmt.Fprintln(stderr, "coordinator event:", err) }, Ready: func() { _, _ = fmt.Fprintln(stdout, "coordinator ready", "https://"+*listen) }}
	if action == "init" {
		if err := clusterapp.InitializeDiLoCoCoordinator(o); err != nil {
			return fail(err)
		}
		_, _ = fmt.Fprintln(stdout, "coordinator contexts initialized; install the service with these same paths and listener")
		return 0
	}
	if action == "install" {
		self, err := serviceExecutable(ctx, binary)
		if err != nil {
			return fail(err)
		}
		m, err := serviceManager()
		if err != nil {
			return fail(err)
		}
		for name, p := range resolved {
			if err := p.Validate(); err != nil {
				return fail(fmt.Errorf("%s: %w", name, err))
			}
		}
		if err := m.Install(ctx, clusterservice.Spec{Role: "coordinator", Binary: self, Arguments: argv}); err != nil {
			return fail(err)
		}
		_, _ = fmt.Fprintln(stdout, "coordinator service installed; use coordinator status and doctor. macOS requires a logged-in GUI session and Local Network permission.")
		return 0
	}
	l, err := net.Listen("tcp", *listen)
	if err != nil {
		return fail(err)
	}
	defer func() { _ = l.Close() }()
	if err := clusterapp.ServeInstalledDiLoCo(ctx, l, o); err != nil {
		return fail(err)
	}
	return 0
}

func serviceExecutable(ctx context.Context, override string) (string, error) {
	self, err := os.Executable()
	if err != nil {
		return "", err
	}
	self, err = filepath.EvalSymlinks(self)
	if err != nil {
		return "", err
	}
	if override != "" {
		if !filepath.IsAbs(override) || filepath.Clean(override) != override {
			return "", fmt.Errorf("cluster-binary must be a canonical absolute path")
		}
		target, err := filepath.EvalSymlinks(override)
		if err != nil || target != self {
			return "", fmt.Errorf("cluster-binary must resolve to this executable")
		}
		self = override
	}
	if runtime.GOOS == "darwin" {
		if err := verifyServiceSignature(ctx, self); err != nil {
			return "", err
		}
	}
	return self, nil
}
