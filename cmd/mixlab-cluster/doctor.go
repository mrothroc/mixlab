package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

type doctorCheck struct {
	Check  string `json:"check"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster doctor", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	agent := f.String("agent-state-dir", "", "inspect an existing local agent installation")
	identity := f.String("principal-state-dir", "", "inspect this enrolled principal; inferred from agent when provided")
	var endpoints endpointFlags
	f.Var(&endpoints, "node", "peer host:port to diagnose; repeatable")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if f.NArg() != 0 {
		return 2
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	checks := []doctorCheck{}
	add := func(name, status, detail string) { checks = append(checks, doctorCheck{name, status, detail}) }
	failed := func(name string, err error) {
		if errors.Is(err, securekeys.ErrUnavailable) {
			add(name, "failed", "keychain_or_backend_unavailable: use the enrolled backend in its authorized session; unlock the login Keychain at the console. No automatic file-backend fallback.")
			return
		}
		d := clusterdiagnostic.Classify(err)
		add(name, "failed", d.Reason+": "+d.Hint)
	}
	if m, err := serviceManager(); err == nil {
		for _, role := range []string{"authority", "agent"} {
			_, err := m.Control(ctx, role, "status")
			if err != nil {
				add(role+"_service", "unknown", "Not installed/running or service manager unavailable; use '"+role+" status' for details.")
			} else {
				add(role+"_service", "ok", "Service manager reports the job. Check trust and peer reachability for readiness.")
			}
		}
	}
	if runtime.GOOS == "darwin" {
		add("local_network_permission", "unknown", "No supported BSD-socket permission query. Allow the signed LaunchAgent once at the console; a logged-in GUI session is required. EHOSTUNREACH can also mean a routing problem.")
		c, stop := context.WithTimeout(ctx, 3*time.Second)
		b, err := exec.CommandContext(c, "/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate").Output()
		stop()
		if err == nil {
			add("application_firewall", "info", strings.TrimSpace(string(b))+" Per-application and network filters may still apply; no settings changed.")
		} else {
			add("application_firewall", "unknown", "Could not read firewall state. Inspect System Settings; no settings changed.")
		}
	} else {
		add("firewall", "unknown", "Check host firewall and SELinux/AppArmor policies. User service installation does not change them.")
	}
	if *agent != "" {
		opts, err := stateOptions(*home, *agent)
		if err != nil {
			failed("agent_state", err)
		} else {
			p, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Agent})
			if err != nil {
				failed("agent_state", err)
			} else {
				i, err := clusterapp.InspectNodeInstallation(p)
				if err != nil {
					failed("agent_installation", err)
				} else {
					if *identity == "" {
						*identity = i.PrincipalDirectory
					}
					_, approvalErr := clusterapp.OpenNodeInstallation(p)
					s, err := nodeagent.Open(p, i.Cluster, i.Node)
					if approvalErr != nil {
						add("executable_approval", "failed", "Executable missing/changed; stop the service and run agent reapprove. Do not recreate the agent state.")
					} else if err != nil {
						add("executable_approval", "failed", "Cannot verify the node approval journal; inspect node state without recreating it.")
					} else if err := s.CheckApprovedWorker(i.WorkerBuild); err != nil {
						add("executable_approval", "failed", "Approval publication incomplete; retry agent reapprove while stopped. Do not recreate the agent state.")
					} else {
						add("executable_approval", "ok", "Executable hashes and capability generation match approval.")
					}
					if err != nil {
						failed("lease_state", err)
					} else if a, err := s.Availability(time.Now()); err != nil {
						failed("lease_state", err)
					} else if !a.Available {
						add("lease_state", "info", "Busy or pending cleanup; reapproval is forbidden.")
					} else {
						add("lease_state", "ok", "No active lease.")
					}
				}
			}
		}
	}
	var nodes []recruitment.NodeStatus
	if *identity != "" {
		nodes = doctorPrincipal(ctx, *home, *identity, endpoints, add, failed)
	} else {
		add("trust", "unknown", "Pass -principal-state-dir or -agent-state-dir to check pinned trust and clock validity.")
		for _, endpoint := range endpoints {
			doctorDial(ctx, endpoint, add, failed)
		}
	}
	add("clock", "info", "Local UTC: "+time.Now().UTC().Format(time.RFC3339)+". Certificate/trust windows are checked when a principal is supplied; absolute peer clock skew cannot be inferred from a failed handshake.")
	result := struct {
		Checks []doctorCheck            `json:"checks"`
		Nodes  []recruitment.NodeStatus `json:"nodes,omitempty"`
	}{checks, nodes}
	if err := json.NewEncoder(stdout).Encode(result); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return 1
	}
	for _, c := range checks {
		if c.Status == "failed" {
			return 1
		}
	}
	for _, n := range nodes {
		if n.Reason != "available" && n.Reason != "busy" {
			return 1
		}
	}
	return 0
}

func doctorDial(ctx context.Context, endpoint string, add func(string, string, string), failed func(string, error)) {
	d := net.Dialer{Timeout: 3 * time.Second}
	c, err := d.DialContext(ctx, "tcp", endpoint)
	if err != nil {
		failed("tcp:"+endpoint, err)
		return
	}
	_ = c.Close()
	add("tcp:"+endpoint, "ok", "TCP reachable only; this does not prove peer identity or trust freshness.")
}

func doctorPrincipal(ctx context.Context, home, identity string, endpoints []string, add func(string, string, string), failed func(string, error)) []recruitment.NodeStatus {
	opts, err := stateOptions(home, identity)
	if err != nil {
		failed("principal", err)
		return nil
	}
	p, err := statehome.Discover(opts, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		failed("principal", err)
		return nil
	}
	s, err := principal.Open(p, time.Now())
	if err != nil {
		failed("principal_or_keychain", err)
		return nil
	}
	defer func() { _ = s.Close() }()
	r, _, err := s.SnapshotReceiver(time.Now())
	if err != nil {
		failed("principal_or_keychain", err)
		return nil
	}
	a, err := trust.PinRoot(r.Root, r.Fingerprint, time.Now())
	if err != nil {
		failed("root_validity", err)
		return nil
	}
	if _, err := trust.VerifySnapshot(a, r.Snapshot, time.Now()); err != nil {
		add("trust_freshness", "failed", "Local pinned trust is stale or invalid. Check clock, authority service and agent refresh log; doctor does not update credentials.")
	} else {
		add("trust_freshness", "ok", "Local signed snapshot is current.")
	}
	if _, _, err := s.Active(time.Now()); err != nil {
		failed("active_identity", err)
	} else {
		add("active_identity", "ok", "Local identity is currently usable.")
	}
	for _, raw := range r.Snapshot.Payload.Endpoints.Payload.URLs {
		u, err := url.Parse(raw)
		if err == nil {
			endpoint := u.Host
			if u.Port() == "" {
				endpoint = net.JoinHostPort(u.Hostname(), "443")
			}
			doctorDial(ctx, endpoint, add, failed)
		}
	}
	if len(endpoints) == 0 {
		return nil
	}
	if r.Role != trust.Controller {
		for _, e := range endpoints {
			doctorDial(ctx, e, add, failed)
		}
		return nil
	}
	lookup, err := clusterapp.NodeLookup(s, time.Now)
	if err != nil {
		failed("node_lookup", err)
		return nil
	}
	hints, err := nodeHints(ctx, "off", endpoints, nil)
	if err != nil {
		failed("node_hints", err)
		return nil
	}
	nodes, err := recruitment.Inventory(ctx, hints, r.Cluster, lookup, time.Now)
	if err != nil {
		failed("node_inventory", err)
	}
	return nodes
}
