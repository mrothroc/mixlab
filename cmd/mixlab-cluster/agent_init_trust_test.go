//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestAgentInitRefreshesExpiredEnrollmentSnapshot(t *testing.T) {
	for _, online := range []bool{false, true} {
		name := "offline"
		if online {
			name = "online"
		}
		t.Run(name, func(t *testing.T) {
			check := func(err error) {
				t.Helper()
				if err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			dir, err := filepath.EvalSymlinks(t.TempDir())
			check(err)
			check(os.Chmod(dir, 0700))
			resolve := func(name string, kind statehome.Kind) statehome.Path {
				p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, name)}, statehome.Context{Kind: kind})
				check(err)
				return p
			}
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			check(err)
			defer func() { _ = listener.Close() }()
			endpoint := "https://" + listener.Addr().String()
			past := time.Now().Add(-20 * time.Minute)
			ids, err := bootstrap.NewIDs()
			check(err)
			ca := resolve("authority", statehome.Authority)
			config := bootstrap.Config{Cluster: ids.Cluster, Authority: ca, Backend: "file", Endpoint: endpoint, Audience: "authority"}
			for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
				config.Principals = append(config.Principals, bootstrap.Target{Role: role, Principal: []string{ids.Authority, ids.Controller, ids.Coordinator}[i], Final: resolve(string(role)+"-principal", statehome.Principal), Staging: resolve(string(role)+"-stage", statehome.Enrollment)})
			}
			_, err = bootstrap.Initialize(ctx, config, past)
			check(err)
			check(clusterapp.InitializeAuthority(ctx, ca, past))
			authority, err := clusterapp.OpenAuthority(ctx, ca, past)
			check(err)
			defer func() { check(authority.Close()) }()
			view, err := authority.Current(ctx, past)
			check(err)
			invite, err := authority.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, view, past)
			check(err)
			defer invite.Clear()
			nodePath := resolve("node", statehome.Principal)
			candidate, err := enrollee.Prepare(ctx, resolve("node-stage", statehome.Enrollment), nodePath, authority.Anchor, trust.Node, "file", past)
			check(err)
			key, envelope, err := candidate.Keys()
			check(err)
			request, err := enrollment.NewRequest(invite, key, envelope, past)
			check(err)
			result, err := authority.Enrollment.Consume(ctx, request, invite.Secret, view, past)
			check(err)
			check(candidate.CompleteProvisioned(ctx, invite, request, result, past))
			check(candidate.Close())
			node, err := principal.Open(nodePath, time.Now())
			check(err)
			defer func() { check(node.Close()) }()
			if _, _, err := node.Active(time.Now()); err == nil {
				t.Fatal("fixture must start with stale trust")
			}
			if online {
				done := make(chan error, 1)
				go func() { done <- clusterapp.ServeAuthority(ctx, listener, authority, time.Now) }()
				defer func() { cancel(); check(<-done) }()
			} else {
				check(listener.Close())
			}
			// Stop before worker probing: this sentinel distinguishes successful
			// CLI trust refresh from rejection at the stale snapshot check.
			missingWorker := filepath.Join(dir, "missing-worker")
			var out, errs bytes.Buffer
			code := runAgentInit(ctx, []string{"-principal-state-dir", nodePath.Dir(), "-agent-state-dir", filepath.Join(dir, "agent"), "-worker-binary", missingWorker, "-agent-relay-listen", "127.0.0.1:27446"}, &out, &errs)
			if code != 1 {
				t.Fatalf("code %d: %s", code, &errs)
			}
			if online {
				if !strings.Contains(errs.String(), "missing-worker") {
					t.Fatalf("did not reach worker validation: %s", &errs)
				}
				_, _, err := node.Active(time.Now())
				check(err)
			} else if !strings.Contains(errs.String(), "startup trust refresh") {
				t.Fatalf("must fail closed with offline authority: %s", &errs)
			}
			if _, err := os.Stat(filepath.Join(dir, "agent")); !os.IsNotExist(err) {
				t.Fatalf("published agent before validation: %v", err)
			}
		})
	}
}
