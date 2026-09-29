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
	"time"

	"github.com/mrothroc/mixlab/internal/buildinfo"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerprobe"
)

func runAgentReapprove(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster agent reapprove", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	state := f.String("agent-state-dir", "", "existing agent directory; identity and history are preserved")
	worker := f.String("worker-binary", "", "absolute newly approved mixlab executable")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if f.NArg() != 0 || !filepath.IsAbs(*worker) {
		return 2
	}
	fail := func(err error) int {
		_, _ = fmt.Fprintln(stderr, "agent reapprove (stop the agent first):", err)
		return 1
	}
	opts, err := stateOptions(*home, *state)
	if err != nil {
		return fail(err)
	}
	p, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		return fail(err)
	}
	w, err := filepath.EvalSymlinks(*worker)
	if err != nil {
		return fail(err)
	}
	g, err := os.Executable()
	if err != nil {
		return fail(err)
	}
	g, err = filepath.EvalSymlinks(g)
	if err != nil {
		return fail(err)
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	i, err := clusterapp.ReapproveNodeInstallation(ctx, p, w, g, func(ctx context.Context, binary, hash string) (workerprobe.Report, error) {
		host, err := workerhost.New(binary, hash)
		if err != nil {
			return workerprobe.Report{}, err
		}
		r, err := initialNodeProbe(ctx, host)
		if err == nil && r.BuildVersion != buildinfo.Report("mixlab") {
			err = fmt.Errorf("worker and cluster binaries must be built from the same version/source state")
		}
		return r, err
	})
	if err != nil {
		return fail(err)
	}
	if err := json.NewEncoder(stdout).Encode(i); err != nil {
		return fail(err)
	}
	return 0
}
