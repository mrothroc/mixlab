package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterservice"
	"github.com/mrothroc/mixlab/statehome"
)

func serviceManager() (clusterservice.Manager, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return clusterservice.Manager{}, err
	}
	home, err = filepath.EvalSymlinks(home)
	return clusterservice.Manager{Platform: runtime.GOOS, Home: home, UID: os.Getuid(), Run: func(ctx context.Context, name string, args ...string) ([]byte, error) {
		cmd := exec.CommandContext(ctx, name, args...)
		cmd.Env = append(os.Environ(), "LC_ALL=C")
		return cmd.CombinedOutput()
	}}, err
}

func isServiceAction(action string) bool {
	switch action {
	case "install", "uninstall", "start", "stop", "status":
		return true
	}
	return false
}

func runService(role, action string, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster "+role+" "+action, flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root for installation discovery")
	launchBinary := f.String("cluster-binary", "", "stable absolute path to this executable (install only; optional symlink such as Homebrew opt)")
	stateFlag, listenFlag, adFlag := "agent-state-dir", "agent-listen", "agent-advertise"
	kind := statehome.Agent
	if role == "authority" {
		stateFlag, listenFlag, adFlag = "cluster-state-dir", "trust-listen", "trust-advertise"
		kind = statehome.Authority
	}
	state := f.String(stateFlag, "", "existing state directory (install only)")
	listen := f.String(listenFlag, "", "explicit listener host:port (install only)")
	advertise := f.String(adFlag, "off", "mdns or off (install only)")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if f.NArg() != 0 {
		return 2
	}
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "service:", err); return 1 }
	m, err := serviceManager()
	if err != nil {
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Second)
	defer cancel()
	if action != "install" {
		if f.NFlag() != 0 {
			_, _ = fmt.Fprintln(stderr, "service control takes no install options")
			return 2
		}
		b, err := m.Control(ctx, role, action)
		_, _ = stdout.Write(b)
		if err != nil {
			return fail(err)
		}
		return 0
	}
	if *advertise != "off" && *advertise != "mdns" {
		return 2
	}
	if role == "agent" && *listen == "" {
		_, _ = fmt.Fprintln(stderr, "agent install requires -agent-listen")
		return 2
	}
	if *listen != "" {
		if _, err := bootstrapEndpoint(*listen); err != nil {
			return fail(err)
		}
	}
	opts, err := stateOptions(*home, *state)
	if err != nil {
		return fail(err)
	}
	p, err := statehome.Discover(opts, statehome.Context{Kind: kind})
	if err != nil {
		return fail(err)
	}
	self, err := os.Executable()
	if err != nil {
		return fail(err)
	}
	self, err = filepath.EvalSymlinks(self)
	if err != nil {
		return fail(err)
	}
	if *launchBinary != "" {
		if !filepath.IsAbs(*launchBinary) || filepath.Clean(*launchBinary) != *launchBinary {
			return fail(fmt.Errorf("cluster-binary must be a canonical absolute path"))
		}
		target, err := filepath.EvalSymlinks(*launchBinary)
		if err != nil || target != self {
			return fail(fmt.Errorf("cluster-binary must resolve to the currently running executable"))
		}
		self = *launchBinary
	}
	if runtime.GOOS == "darwin" {
		if err := verifyServiceSignature(ctx, self); err != nil {
			return fail(err)
		}
	}
	if role == "agent" {
		i, err := clusterapp.OpenNodeInstallation(p)
		if err != nil {
			return fail(err)
		}
		executable, err := filepath.EvalSymlinks(self)
		if err != nil || executable != i.GuardianBinary {
			return fail(fmt.Errorf("install using the approved cluster binary, or stop and reapprove first"))
		}
	}
	argv := []string{role}
	if role == "authority" {
		argv = append(argv, "serve")
	}
	argv = append(argv, "-"+stateFlag, p.Dir(), "-"+adFlag, *advertise)
	if *listen != "" {
		argv = append(argv, "-"+listenFlag, *listen)
	}
	if err := m.Install(ctx, clusterservice.Spec{Role: role, Binary: self, Arguments: argv}); err != nil {
		return fail(err)
	}
	_, _ = fmt.Fprintf(stdout, "%s service installed. Use '%s status' and 'doctor' to check readiness.\n", role, role)
	if runtime.GOOS == "darwin" {
		_, _ = fmt.Fprintln(stdout, "Requires a logged-in GUI session. Click Allow for Local Network access on this Mac. Logs: ~/.mixlab/services/"+role+"/service.log")
	} else {
		_, _ = fmt.Fprintln(stdout, "Logged-out operation requires administrator-enabled systemd lingering; installation does not change it.")
	}
	return 0
}

func verifyServiceSignature(ctx context.Context, path string) error {
	// Validate a Developer ID requirement, not merely the presence of a cdhash.
	const requirement = `=anchor apple generic and identifier "com.mixlab.mixlab-cluster" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists`
	b, err := exec.CommandContext(ctx, "/usr/bin/codesign", "--verify", "--strict", "-R", requirement, path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("macOS services require the Developer ID-signed Mixlab package (not the source-built formula): %w: %s", err, b)
	}
	return nil
}

func runServiceProcess(args []string, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster service-run", flag.ContinueOnError)
	f.SetOutput(stderr)
	dir := f.String("service-dir", "", "private installed service directory")
	if err := f.Parse(args); err != nil {
		return 2
	}
	if f.NArg() != 0 || !filepath.IsAbs(*dir) {
		return 2
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: *dir}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	s, err := clusterservice.Read(p)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	log := &clusterservice.TailLog{Path: p}
	if _, err := fmt.Fprintf(log, "\nservice start %s\n", time.Now().UTC().Format(time.RFC3339)); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if s.Role == "agent" {
		return runAgentSupervised(ctx, s.Arguments[1:], log, log, func(attempt func() int) int {
			return runServiceRetry(ctx, log, attempt)
		})
	}
	return runServiceRetry(ctx, log, func() int {
		return runAuthorityServeContext(ctx, s.Arguments[2:], log, log)
	})
}

// The OS service owns cancellation; retries never create a second signal
// subscription that could miss a stop between attempts.
func runServiceRetry(ctx context.Context, log io.Writer, attempt func() int) int {
	for {
		if ctx.Err() != nil {
			return 0
		}
		code := attempt()
		if ctx.Err() != nil {
			return 0
		}
		// Stay alive long enough for macOS to show the LNP alert after a
		// denied startup dial. A one-shot process may never get a prompt.
		if _, err := fmt.Fprintf(log, "service exited %d at %s; retry in 30s (check network permission and trust; stop the service before agent reapprove)\n", code, time.Now().UTC().Format(time.RFC3339)); err != nil {
			return 1
		}
		timer := time.NewTimer(30 * time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return 0
		case <-timer.C:
		}
	}
}
