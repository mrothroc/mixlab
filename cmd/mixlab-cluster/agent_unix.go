//go:build darwin || linux

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
	"path/filepath"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerprobe"
)

func runAgent(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	if len(args) > 0 && args[0] == "init" {
		return runAgentInit(ctx, args[1:], stdout, stderr)
	}
	if len(args) > 0 && args[0] == "reapprove" {
		return runAgentReapprove(ctx, args[1:], stdout, stderr)
	}
	return runAgentContext(ctx, args, stdout, stderr)
}

func runAgentContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	return runAgentSupervised(ctx, args, stdout, stderr, func(attempt func() int) int { return attempt() })
}

// The supervisor remains inside the installation lock across failed starts and
// retry delays. Reapproval cannot bless new bytes while an old image is alive.
func runAgentSupervised(ctx context.Context, args []string, stdout, stderr io.Writer, supervise func(func() int) int) int {
	f := flag.NewFlagSet("mixlab-cluster agent", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	state := f.String("agent-state-dir", "", "initialized agent directory; never created by startup")
	listen := f.String("agent-listen", "127.0.0.1:7445", "explicit local IP:port for managed control TLS")
	advertise := f.String("agent-advertise", "off", "mdns or off; advertisements are untrusted hints")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "agent:", err); return 1 }
	if f.NArg() != 0 || (*advertise != "off" && *advertise != "mdns") {
		return 2
	}
	if _, err := bootstrapEndpoint(*listen); err != nil {
		return fail(err)
	}
	opts, err := stateOptions(*home, *state)
	if err != nil {
		return fail(err)
	}
	path, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		return fail(err)
	}
	code := 1
	err = path.WithProcessLock(ctx, clusterapp.NodeServiceLock, func() error {
		code = supervise(func() int { return serveInstalledAgent(ctx, path, *listen, *advertise, stdout, stderr) })
		return nil
	})
	if err != nil {
		return fail(err)
	}
	return code
}

func serveInstalledAgent(ctx context.Context, path statehome.Path, listen, advertise string, stdout, stderr io.Writer) int {
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "agent:", err); return 1 }
	i, err := clusterapp.OpenNodeInstallation(path)
	if err != nil {
		return fail(err)
	}
	self, err := os.Executable()
	if err != nil {
		return fail(err)
	}
	self, err = filepath.EvalSymlinks(self)
	if err != nil || self != i.GuardianBinary {
		return fail(fmt.Errorf("agent must run the locally approved cluster executable"))
	}
	pp, err := statehome.Resolve(statehome.Options{ExactDir: i.PrincipalDirectory}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return fail(err)
	}
	p, err := principal.Open(pp, time.Now())
	if err != nil {
		return fail(err)
	}
	defer func() { _ = p.Close() }()
	if err := clusterapp.RefreshNodeStartup(ctx, p, time.Now); err != nil {
		return fail(fmt.Errorf("startup trust refresh: %w", err))
	}
	identity, _, err := p.Active(time.Now())
	if err != nil {
		return fail(err)
	}
	if identity.Cluster != i.Cluster || identity.Principal != i.Node {
		return fail(fmt.Errorf("installed node identity changed"))
	}
	current := clusterapp.NodeCurrentTrust(p)
	anchor, _, err := current(ctx, time.Now())
	if err != nil {
		return fail(err)
	}
	store, runtime, credentialRoot, err := clusterapp.OpenNodeRuntime(path, i, anchor)
	if err != nil {
		return fail(err)
	}
	host, err := workerhost.New(i.WorkerBinary, i.WorkerBuild)
	if err != nil {
		return fail(err)
	}
	probePath, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(path.Dir(), "probe")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return fail(err)
	}
	if err := probePath.Validate(); err != nil {
		return fail(err)
	}
	probe := func(ctx context.Context) (workerprobe.Report, error) { return host.Probe(ctx, probePath) }
	workload, err := clusterapp.NodeWorkloadPorts(p, credentialRoot, time.Now)
	if err != nil {
		return fail(err)
	}
	// ServeNode drains HTTP before canceling execution. Keep application relay
	// lifetime alive through that ordered shutdown, then cancel on return.
	life, done := context.WithCancel(context.WithoutCancel(ctx))
	defer done()
	jobs, err := clusterapp.NewNodeJobs(life, clusterapp.NodeJobOptions{Store: store, Node: i.Node, Anchor: anchor, CredentialRoot: credentialRoot, Workload: workload, RelayAddress: i.RelayAddress, CurrentTrust: current, Clock: time.Now,
		Output:    runtime.Output,
		InputRoot: path,
		Preparation: nodeagent.PreparationPorts{Clock: time.Now, Probe: probe, DatasetID: data.DistributedDatasetIdentityContext, ArtifactPresent: func(context.Context, nodejob.ArtifactRef) error {
			return fmt.Errorf("artifact requires the authenticated checkpoint input path")
		}},
	})
	if err != nil {
		return fail(err)
	}
	l, err := net.Listen("tcp", listen)
	if err != nil {
		return fail(err)
	}
	defer func() { _ = l.Close() }()
	ad, err := advertiseListener(ctx, advertise, l, discovery.Node, i.Node, nodeDiscoveryClaims(i.Cluster, i.Node))
	if err != nil {
		return fail(err)
	}
	defer func() { _ = ad.Close() }()
	event := func(err error) { _, _ = fmt.Fprintln(stderr, "agent event:", err) }
	maintenance := func(ctx context.Context) error {
		life, cancel := context.WithCancel(ctx)
		defer cancel()
		trustDone := make(chan error, 1)
		go func() { trustDone <- clusterapp.MaintainNodeTrust(life, pp, p, time.Now, event); cancel() }()
		probeErr := maintainIdleProbe(life, store, probe, event)
		cancel()
		return errors.Join(probeErr, <-trustDone)
	}
	err = clusterapp.ServeNode(ctx, l, clusterapp.NodeServerOptions{Store: store, Jobs: jobs,
		Snapshots: &clusterapp.NodeSnapshots{Principal: p, Clock: time.Now},
		Policy:    func() (*managedtls.Policy, error) { return clusterapp.NodePrincipalPolicy(p, time.Now) }, Maintenance: maintenance,
		Ready: func() {
			if err := json.NewEncoder(stdout).Encode(struct{ Node, Listen, Status string }{i.Node, l.Addr().String(), "ready"}); err != nil {
				event(err)
			}
		},
		Execution: clusterapp.NodeExecutionPorts{Open: runtime.Open, Authorize: jobs.Authorize, Cleanup: jobs.Cleanup, Clock: time.Now, Event: event},
	})
	if err != nil {
		return fail(err)
	}
	return 0
}

func maintainIdleProbe(ctx context.Context, store *nodeagent.Store, probe func(context.Context) (workerprobe.Report, error), event func(error)) error {
	tick := time.NewTicker(time.Minute)
	defer tick.Stop()
	for {
		if err := store.RefreshIdleProbe(ctx, probe, time.Now); err != nil && ctx.Err() == nil {
			event(err)
		}
		select {
		case <-ctx.Done():
			return nil
		case <-tick.C:
		}
	}
}
