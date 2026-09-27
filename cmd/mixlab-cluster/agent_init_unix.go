//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func runAgentInit(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster agent init", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	state := f.String("agent-state-dir", "", "new private agent directory; existing state is never reset")
	identity := f.String("principal-state-dir", "", "existing enrolled node identity directory")
	worker := f.String("worker-binary", "", "administrator-approved absolute mixlab executable")
	relay := f.String("agent-relay-listen", "", "explicit local IP:port for encrypted job transport")
	name := f.String("name", "mixlab-node", "local display name")
	runtime := f.Int("max-runtime-seconds", 86400, "maximum job wall time")
	cpu := f.Int64("max-cpu-seconds", 172800, "maximum monitored aggregate child CPU time")
	memory := f.Uint64("max-memory-bytes", 4<<30, "maximum monitored child resident memory")
	disk := f.Uint64("max-disk-bytes", 4<<30, "maximum monitored private attempt disk usage")
	logs := f.Uint64("max-log-bytes", 16<<20, "maximum retained worker log bytes")
	var datasets repeatedFlag
	f.Var(&datasets, "dataset", "local NAME=absolute-shard-glob registration; repeatable")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "agent init:", err); return 1 }
	if f.NArg() != 0 || *identity == "" || !filepath.IsAbs(*worker) || *relay == "" {
		_, _ = fmt.Fprintln(stderr, "agent init requires principal-state-dir, absolute worker-binary and agent-relay-listen")
		return 2
	}
	limits := nodejob.Limits{RuntimeSeconds: *runtime, CPUSeconds: *cpu, MemoryBytes: *memory, DiskBytes: *disk, LogBytes: *logs}
	if err := limits.Validate(); err != nil {
		return fail(err)
	}
	po, err := stateOptions(*home, *identity)
	if err != nil {
		return fail(err)
	}
	pp, err := statehome.Resolve(po, statehome.Context{Kind: statehome.Principal})
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
	current, _, err := p.Active(time.Now())
	if err != nil {
		return fail(err)
	}
	if current.Role != trust.Node {
		return fail(fmt.Errorf("agent setup requires an enrolled node, not %s", current.Role))
	}
	options, err := stateOptions(*home, *state)
	if err != nil {
		return fail(err)
	}
	path, err := statehome.Resolve(options, statehome.Context{Kind: statehome.Agent, ClusterID: current.Cluster, ID: current.Principal})
	if err != nil {
		return fail(err)
	}
	workerPath, err := filepath.EvalSymlinks(*worker)
	if err != nil {
		return fail(err)
	}
	guardian, err := os.Executable()
	if err != nil {
		return fail(err)
	}
	guardian, err = filepath.EvalSymlinks(guardian)
	if err != nil {
		return fail(err)
	}
	build, err := workerjob.FileDigest(workerPath)
	if err != nil {
		return fail(err)
	}
	guardianBuild, err := workerjob.FileDigest(guardian)
	if err != nil {
		return fail(err)
	}
	host, err := workerhost.New(workerPath, build)
	if err != nil {
		return fail(err)
	}
	probe, err := initialNodeProbe(ctx, host)
	if err != nil {
		return fail(err)
	}
	profile := nodeagent.Profile{Version: nodeagent.ProfileVersion, Node: current.Principal, DisplayName: *name, Generation: 1, Probe: probe, ProbeObservedAt: time.Now().Unix(), Limits: limits, Datasets: []nodeagent.LocalDataset{}}
	profile.TransportEndpoint = *relay
	for _, item := range datasets {
		selector, pattern, ok := strings.Cut(item, "=")
		if !ok || !filepath.IsAbs(pattern) || filepath.Clean(pattern) != pattern {
			return fail(fmt.Errorf("dataset requires NAME=canonical-absolute-glob"))
		}
		id, err := data.DistributedDatasetIdentityContext(ctx, pattern)
		if err != nil {
			return fail(err)
		}
		profile.Datasets = append(profile.Datasets, nodeagent.LocalDataset{Dataset: nodeagent.Dataset{Selector: selector, ID: id}, TrainPattern: pattern})
	}
	i := clusterapp.NodeInstallation{Cluster: current.Cluster, Node: current.Principal, PrincipalDirectory: pp.Dir(), WorkerBinary: workerPath, WorkerBuild: build, GuardianBinary: guardian, GuardianBuild: guardianBuild, RelayAddress: *relay}
	if err := clusterapp.InitializeNodeInstallation(ctx, path, i, profile); err != nil {
		return fail(err)
	}
	if err := json.NewEncoder(stdout).Encode(struct{ Node, State string }{current.Principal, path.Dir()}); err != nil {
		return fail(err)
	}
	return 0
}

func initialNodeProbe(ctx context.Context, host *workerhost.Supervisor) (workerprobe.Report, error) {
	dir, err := os.MkdirTemp("", "mixlab-node-probe-")
	if err != nil {
		return workerprobe.Report{}, err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	dir, err = filepath.EvalSymlinks(dir)
	if err != nil {
		return workerprobe.Report{}, err
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return workerprobe.Report{}, err
	}
	return host.Probe(ctx, p)
}
