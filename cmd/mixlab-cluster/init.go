package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
)

func runInit(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("mixlab-cluster init", flag.ContinueOnError)
	flags.SetOutput(stderr)
	home := flags.String("state-home", "", "state root (then MIXLAB_STATE_HOME, then ~/.mixlab)")
	ca := flags.String("cluster-state-dir", "", "exact cluster authority directory")
	authority := flags.String("authority-principal-state-dir", "", "exact initial authority TLS principal directory")
	controller := flags.String("controller-principal-state-dir", "", "exact initial controller principal directory")
	coordinator := flags.String("coordinator-principal-state-dir", "", "exact initial coordinator principal directory")
	listen := flags.String("trust-listen", "127.0.0.1:7443", "future authority host:port; init opens no listener")
	advertise := flags.String("trust-advertise", "off", "off only during experimental local initialization")
	backend := flags.String("key-backend", "", "new identity key storage: keychain (macOS default) or file")
	recover := flags.Bool("recover", false, "recover the persisted initialization; requires -cluster-state-dir and no new options")
	flags.Usage = func() {
		_, _ = fmt.Fprintln(stderr, "Usage: mixlab-cluster init [options]\n"+developmentNotice)
		flags.PrintDefaults()
	}
	if err := flags.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "mixlab-cluster init:", err); return 2 }
	if flags.NArg() != 0 {
		return invalid(fmt.Errorf("unexpected positional argument %q", flags.Arg(0)))
	}
	wd, err := os.Getwd()
	if err != nil {
		return invalid(err)
	}
	var report bootstrap.Report
	var authorityPath statehome.Path
	if *recover {
		if *ca == "" {
			return invalid(fmt.Errorf("-recover requires -cluster-state-dir"))
		}
		var forbidden string
		flags.Visit(func(f *flag.Flag) {
			if f.Name != "recover" && f.Name != "cluster-state-dir" {
				forbidden = f.Name
			}
		})
		if forbidden != "" {
			return invalid(fmt.Errorf("-recover uses persisted options; -%s is not allowed", forbidden))
		}
		authorityPath, err = statehome.Resolve(statehome.Options{ExactDir: *ca, WorkingDir: wd}, statehome.Context{Kind: statehome.Authority})
		if err != nil {
			return invalid(err)
		}
		report, err = bootstrap.Recover(context.Background(), authorityPath, time.Now())
	} else {
		if *advertise != "off" {
			return invalid(fmt.Errorf("-trust-advertise only supports off until the authority service is available"))
		}
		endpoint, e := bootstrapEndpoint(*listen)
		if e != nil {
			return invalid(e)
		}
		userHome, e := os.UserHomeDir()
		if e != nil {
			return invalid(e)
		}
		ids, e := bootstrap.NewIDs()
		if e != nil {
			return invalid(e)
		}
		base := statehome.Options{Flag: *home, Env: os.Getenv("MIXLAB_STATE_HOME"), UserHome: userHome, WorkingDir: wd}
		resolve := func(exact string, c statehome.Context) (statehome.Path, error) {
			o := base
			o.ExactDir = exact
			return statehome.Resolve(o, c)
		}
		authorityPath, err = resolve(*ca, statehome.Context{Kind: statehome.Authority, ClusterID: ids.Cluster})
		if err != nil {
			return invalid(err)
		}
		c := bootstrap.Config{Cluster: ids.Cluster, Authority: authorityPath, Backend: *backend, Endpoint: endpoint, Audience: "mixlab-trust"}
		for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
			id := []string{ids.Authority, ids.Controller, ids.Coordinator}[i]
			p, e := resolve([]string{*authority, *controller, *coordinator}[i], statehome.Context{Kind: statehome.Principal, ClusterID: ids.Cluster, Role: string(role), ID: id})
			if e != nil {
				return invalid(e)
			}
			stage, e := resolve("", statehome.Context{Kind: statehome.Enrollment, ID: id})
			if e != nil {
				return invalid(e)
			}
			c.Principals = append(c.Principals, bootstrap.Target{Role: role, Principal: id, Final: p, Staging: stage})
		}
		report, err = bootstrap.Initialize(context.Background(), c, time.Now())
	}
	if err == nil {
		err = clusterapp.InitializeAuthority(context.Background(), authorityPath, time.Now())
	}
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "mixlab-cluster init: %v\nAuthority directory: %s\nIf initialization was interrupted, use init -recover -cluster-state-dir with this directory; do not replace keys.\n", err, authorityPath.Dir())
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		_, _ = fmt.Fprintln(stderr, "initialization succeeded but reporting failed:", err)
		return 1
	}
	return 0
}

// Shared by -trust-listen, -enrollment-listen and -agent-listen, so the
// message names no single flag: reporting the wrong one sends the reader to
// a flag they did not pass.
func bootstrapEndpoint(address string) (string, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return "", fmt.Errorf("listen address requires a concrete host:port: %w", err)
	}
	n, err := strconv.Atoi(port)
	ip := net.ParseIP(host)
	if err != nil || n < 1 || n > 65535 || host == "" || (ip != nil && ip.IsUnspecified()) || strings.ContainsAny(host, " /?#@%\\\t\n\r") {
		return "", fmt.Errorf("listen address requires a concrete reachable host and port 1..65535; a wildcard bind such as 0.0.0.0 is refused")
	}
	return "https://" + net.JoinHostPort(host, strconv.Itoa(n)), nil
}
