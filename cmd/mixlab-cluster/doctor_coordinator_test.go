package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestDoctorCoordinatorTargetAndHelp(t *testing.T) {
	id := strings.Repeat("a", 32)
	for endpoint, want := range map[string]string{"https://host": "host:443", "https://host:7446": "host:7446", "https://[::1]:7446": "[::1]:7446"} {
		if got, err := doctorCoordinatorTarget(endpoint, id); err != nil || got != want {
			t.Fatalf("%s: %s %v", endpoint, got, err)
		}
	}
	for _, endpoint := range []string{"", "http://host", "https://host/path", "https://user:secret@host", "https://host?secret", "https://host?", "https://host#fragment", "https://host:0", "https://host:65536"} {
		var out, stderr bytes.Buffer
		if code := runDoctor([]string{"-coordinator-endpoint", endpoint, "-coordinator-id", id}, &out, &stderr); code != 2 || out.Len() != 0 {
			t.Fatalf("accepted %q: %d %s", endpoint, code, &stderr)
		}
		if strings.Contains(stderr.String(), "secret") {
			t.Fatal("endpoint secrets echoed")
		}
	}
	for _, id := range []string{"", "a", strings.Repeat("A", 32), strings.Repeat("g", 32)} {
		if _, err := doctorCoordinatorTarget("https://host", id); err == nil {
			t.Fatalf("accepted coordinator ID %q", id)
		}
	}
	var out, stderr bytes.Buffer
	if code := runDoctor([]string{"-help"}, &out, &stderr); code != 0 || !strings.Contains(stderr.String(), "-coordinator-endpoint") || !strings.Contains(stderr.String(), "-coordinator-id") {
		t.Fatalf("help %d: %s", code, &stderr)
	}
}

func TestDoctorCoordinatorObservationDoesNotInferPermissionDenial(t *testing.T) {
	for _, tt := range []struct {
		err    error
		reason string
	}{
		{nil, "Read-only Info request"},
		{syscall.EHOSTUNREACH, "route_or_permission_denied"},
		{syscall.ECONNREFUSED, "connection_refused"},
		{syscall.EACCES, "permission_denied"},
		{context.DeadlineExceeded, "timeout"},
		{tls.AlertError(42), "tls_rejected"},
		{clusterdiagnostic.ErrLocalIdentity, "local_identity_unavailable"},
		{securekeys.ErrUnavailable, "keychain_or_backend_unavailable"},
		{errors.New("remote secret /private-key"), "unavailable_or_unauthenticated"},
	} {
		t.Run(tt.reason, func(t *testing.T) {
			var checks []doctorCheck
			add := func(name, status, detail string) { checks = append(checks, doctorCheck{name, status, detail}) }
			doctorCoordinatorInfo(context.Background(), func(ctx context.Context) (clusterapp.DiLoCoInfo, error) {
				deadline, ok := ctx.Deadline()
				if !ok || time.Until(deadline) > 5*time.Second {
					t.Fatal("unbounded diagnostic")
				}
				if tt.err != nil {
					return clusterapp.DiLoCoInfo{}, fmt.Errorf("secret: %w", tt.err)
				}
				return clusterapp.DiLoCoInfo{}, nil
			}, add, func(name string, err error) { add(name, "failed", doctorFailure(err)) })
			if len(checks) != 1 || checks[0].Check != "coordinator_tls_trust" || !strings.Contains(checks[0].Detail, tt.reason) || strings.Contains(checks[0].Detail, "secret") {
				t.Fatal(checks)
			}
			if tt.err != nil && checks[0].Status != "failed" || tt.err == nil && checks[0].Status != "ok" {
				t.Fatal(checks)
			}
			if tt.err != nil && !errors.Is(tt.err, securekeys.ErrUnavailable) && !strings.Contains(checks[0].Detail, "does not establish") {
				t.Fatal("failure overclaims permission evidence", checks)
			}
		})
	}
}

func TestDoctorCoordinatorAuthenticatedLoopback(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	_, args, report := coordinatorCLIFixture(t)
	args[len(args)-1] = availableAddress(t)
	var initOut, initErr bytes.Buffer
	if code := runCoordinatorContext(context.Background(), "init", args, &initOut, &initErr); code != 0 {
		t.Fatalf("init %d: %s", code, &initErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	reader, writer := io.Pipe()
	defer func() { _ = reader.Close() }()
	done := make(chan int, 1)
	go func() {
		done <- runCoordinatorContext(ctx, "serve", args, writer, io.Discard)
		_ = writer.Close()
	}()
	defer func() {
		cancel()
		if code := <-done; code != 0 {
			t.Errorf("serve exit %d", code)
		}
	}()
	line, err := bufio.NewReader(reader).ReadString('\n')
	if err != nil || !strings.Contains(line, "coordinator ready") {
		t.Fatalf("readiness: %q %v", line, err)
	}
	controller := fixturePrincipal(report, trust.Controller)
	coordinator := fixturePrincipal(report, trust.Coordinator)
	for _, tt := range []struct{ name, identity, id, check, status string }{
		{"authenticated", controller.Directory, coordinator.Principal, "coordinator_tls_trust", "ok"},
		{"wrong-peer", controller.Directory, strings.Repeat("f", 32), "coordinator_tls_trust", "failed"},
		{"no-identity", "", coordinator.Principal, "coordinator_tls_trust", "unknown"},
		{"wrong-role", coordinator.Directory, coordinator.Principal, "coordinator_local_identity", "failed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			checks := map[string]doctorCheck{}
			add := func(name, status, detail string) { checks[name] = doctorCheck{name, status, detail} }
			doctorCoordinator(ctx, "", tt.identity, "https://"+args[len(args)-1], tt.id, add, func(name string, err error) { add(name, "failed", doctorFailure(err)) })
			if checks[tt.check].Status != tt.status || checks["tcp:"+args[len(args)-1]].Status != "ok" {
				t.Fatal(checks)
			}
		})
	}
	t.Run("full-doctor-controller", func(t *testing.T) {
		check := func(err error) {
			t.Helper()
			if err != nil {
				t.Fatal(err)
			}
		}
		p, err := statehome.Resolve(statehome.Options{ExactDir: controller.Directory}, statehome.Context{Kind: statehome.Principal})
		check(err)
		s, err := principal.Open(p, time.Now())
		check(err)
		defer func() { check(s.Close()) }()
		view, err := s.View(time.Now())
		check(err)
		endpoint, err := url.Parse(view.Snapshot.Payload.Endpoints.Payload.URLs[0])
		check(err)
		listener, err := net.Listen("tcp", endpoint.Host)
		check(err)
		defer func() { _ = listener.Close() }()
		ca, err := statehome.Resolve(statehome.Options{ExactDir: report.AuthorityDir}, statehome.Context{Kind: statehome.Authority})
		check(err)
		a, err := clusterapp.OpenAuthority(ctx, ca, time.Now())
		check(err)
		defer func() { check(a.Close()) }()
		authorityCtx, stop := context.WithCancel(ctx)
		done := make(chan error, 1)
		go func() { done <- clusterapp.ServeAuthority(authorityCtx, listener, a, time.Now) }()
		defer func() { stop(); check(<-done) }()
		doctorArgs := []string{"-principal-state-dir", controller.Directory, "-coordinator-endpoint", "https://" + args[len(args)-1], "-coordinator-id", coordinator.Principal}
		run := func(wantCode int) map[string]doctorCheck {
			t.Helper()
			var out, stderr bytes.Buffer
			if code := runDoctor(doctorArgs, &out, &stderr); code != wantCode {
				t.Fatalf("doctor exit %d, want %d: %s %s", code, wantCode, &out, &stderr)
			}
			var response struct {
				Checks []doctorCheck `json:"checks"`
			}
			check(json.Unmarshal(out.Bytes(), &response))
			checks := make(map[string]doctorCheck)
			for _, c := range response.Checks {
				checks[c.Check] = c
				if c.Check == "principal_or_keychain" || wantCode == 0 && c.Status == "failed" {
					t.Fatalf("unexpected failure: %+v", c)
				}
			}
			return checks
		}
		checks := run(0)
		for _, name := range []string{"trust_freshness", "active_identity", "coordinator_tls_trust", "tcp:" + endpoint.Host} {
			if checks[name].Status != "ok" {
				t.Fatalf("missing successful %s: %v", name, checks)
			}
		}
		// A readable, current snapshot must not substitute for an active identity.
		snapshot, err := a.Snapshots.Revoke(ctx, "principal", controller.Principal, "prospective", "doctor regression", time.Now())
		check(err)
		check(s.Refresh(ctx, snapshot, time.Now()))
		checks = run(1)
		if checks["trust_freshness"].Status != "ok" || checks["active_identity"].Status != "failed" || checks["coordinator_local_identity"].Status != "failed" {
			t.Fatalf("revoked identity was not rejected: %v", checks)
		}
	})
}

func TestDoctorReportsLocalCoordinatorServiceWithoutCreatingState(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	var out, stderr bytes.Buffer
	if code := runDoctor(nil, &out, &stderr); code != 0 {
		t.Fatal(code, &stderr)
	}
	var response struct {
		Checks []doctorCheck `json:"checks"`
	}
	if err := json.Unmarshal(out.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	for _, c := range response.Checks {
		if c.Check == "coordinator_service" && c.Status == "unknown" {
			return
		}
	}
	t.Fatal("missing local coordinator service observation", &out)
}
