package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust/principal"
)

type endpointFlags []string

func (s *endpointFlags) String() string { return fmt.Sprint([]string(*s)) }
func (s *endpointFlags) Set(v string) error {
	if len(*s) >= discovery.MaxHints {
		return fmt.Errorf("too many node endpoints")
	}
	if _, err := (discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: {v}}}).Browse(context.Background(), discovery.Node); err != nil {
		return err
	}
	*s = append(*s, v)
	return nil
}

func runNodes(args []string, stdout, stderr io.Writer) int {
	return runNodesContext(context.Background(), args, stdout, stderr)
}

func runNodesContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster nodes", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	state := f.String("principal-state-dir", "", "enrolled controller principal directory")
	legacyState := f.String("controller-state-dir", "", "alias for principal-state-dir")
	ca := f.String("cluster-state-dir", "", "local authority to serve temporarily; omit for an existing service")
	endpoint := f.String("authority-endpoint", "", "root-signed authority URL; defaults to its sole signed URL")
	pool := f.String("pool", "local", "local only")
	mode := f.String("discover", "off", "mdns or off; discovery is address hints only")
	var endpoints endpointFlags
	f.Var(&endpoints, "node", "explicit agent host:port (repeatable)")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if f.NArg() != 0 || *pool != "local" || (*state != "" && *legacyState != "") || (*mode != "off" && *mode != "mdns") || (*mode == "off" && len(endpoints) == 0) {
		_, _ = fmt.Fprintln(stderr, "nodes: use -node host:port or explicitly enable -discover mdns")
		return 2
	}
	fail := func(err error) int { _, _ = fmt.Fprintln(stderr, "nodes:", err); return 1 }
	if *state == "" {
		*state = *legacyState
	}
	err := withController(ctx, *home, *state, *ca, *endpoint, stderr, func(ctx context.Context, p *principal.Store, _ string) error {
		identity, _, err := p.Active(time.Now())
		if err != nil {
			return err
		}
		lookup, err := clusterapp.NodeLookup(p, time.Now)
		if err != nil {
			return err
		}
		hints, err := nodeHints(ctx, *mode, endpoints, discovery.MDNS{})
		if err != nil {
			return err
		}
		rows, err := recruitment.Inventory(ctx, hints, identity.Cluster, lookup, time.Now)
		if err != nil {
			return err
		}
		return json.NewEncoder(stdout).Encode(rows)
	})
	if err != nil {
		return fail(err)
	}
	return 0
}

func nodeHints(ctx context.Context, mode string, endpoints []string, mdns discovery.Provider) ([]discovery.Hint, error) {
	explicit := discovery.Explicit{Addresses: map[discovery.Service][]string{discovery.Node: endpoints}}
	hints, err := explicit.Browse(ctx, discovery.Node)
	if err != nil {
		return nil, err
	}
	switch mode {
	case "off":
		return hints, nil
	case "mdns":
		if mdns == nil {
			return nil, fmt.Errorf("mDNS provider required")
		}
		query, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		more, err := mdns.Browse(query, discovery.Node)
		if err != nil {
			return nil, err
		}
		if len(hints)+len(more) > discovery.MaxHints {
			return nil, fmt.Errorf("too many node hints; use explicit endpoints")
		}
		return append(hints, more...), nil
	default:
		return nil, fmt.Errorf("discovery must be mdns or off")
	}
}
