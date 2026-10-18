package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/internal/clusterservice"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/principal"
)

func coordinatorCLIArgs(root string) []string {
	var args []string
	for _, name := range clusterservice.CoordinatorPathFlags() {
		args = append(args, "-"+name, filepath.Join(root, name))
	}
	return append(args, "-coordinator-listen", "127.0.0.1:7446")
}

func TestCoordinatorCLIConfigurationValidation(t *testing.T) {
	for _, action := range []string{"init", "serve", "install"} {
		t.Run(action, func(t *testing.T) {
			root := t.TempDir()
			good := coordinatorCLIArgs(root)
			for i := 0; i < len(good); i += 2 {
				args := append(append([]string{}, good[:i]...), good[i+2:]...)
				var out, stderr bytes.Buffer
				if code := runCoordinatorContext(context.Background(), action, args, &out, &stderr); code != 1 || out.Len() != 0 || stderr.Len() == 0 {
					t.Fatalf("missing %s: %d %s %s", good[i], code, &out, &stderr)
				}
			}
			for _, value := range []string{"relative", root + "/../elsewhere", good[3]} {
				args := append([]string{}, good...)
				args[1] = value
				var out, stderr bytes.Buffer
				if code := runCoordinatorContext(context.Background(), action, args, &out, &stderr); code != 1 {
					t.Fatalf("path %q: %d %s", value, code, &stderr)
				}
			}
			for _, listen := range []string{"0.0.0.0:7446", "[::]:7446", ":7446", "host:0", "host:65536", "https://host:7446"} {
				args := append([]string{}, good...)
				args[len(args)-1] = listen
				var out, stderr bytes.Buffer
				if code := runCoordinatorContext(context.Background(), action, args, &out, &stderr); code != 1 {
					t.Fatalf("listener %q: %d %s", listen, code, &stderr)
				}
			}
			entries, err := os.ReadDir(root)
			if err != nil || len(entries) != 0 {
				t.Fatalf("invalid configuration created state: %v %v", entries, err)
			}
		})
	}
}

func coordinatorCLIFixture(t *testing.T) (string, []string, bootstrap.Report) {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(root, 0700); err != nil {
		t.Fatal(err)
	}
	args := coordinatorCLIArgs(root)
	var out, stderr bytes.Buffer
	if code := runInit([]string{"-state-home", filepath.Join(root, "bootstrap"), "-key-backend", "file", "-trust-listen", availableAddress(t), "-coordinator-principal-state-dir", args[1], "-controller-principal-state-dir", args[3]}, &out, &stderr); code != 0 {
		t.Fatalf("bootstrap %d: %s", code, &stderr)
	}
	var report bootstrap.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	return root, args, report
}

func TestCoordinatorCLIInitIsIdempotentAndBoundToConfiguration(t *testing.T) {
	root, args, _ := coordinatorCLIFixture(t)
	t.Setenv("MIXLAB_STATE_HOME", filepath.Join(root, "unused"))
	for i := 0; i < 2; i++ {
		var out, stderr bytes.Buffer
		if code := runCoordinatorContext(context.Background(), "init", args, &out, &stderr); code != 0 || !strings.Contains(out.String(), "initialized") {
			t.Fatalf("init %d: %s", code, &stderr)
		}
	}
	for _, name := range clusterservice.CoordinatorPathFlags()[2:] {
		if _, err := os.Stat(filepath.Join(root, name, "coordinator-owner.json")); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "unused")); !os.IsNotExist(err) {
		t.Fatal("explicit roots used environment", err)
	}
	changed := append([]string{}, args...)
	changed[len(changed)-1] = "127.0.0.1:7447"
	var out, stderr bytes.Buffer
	if code := runCoordinatorContext(context.Background(), "init", changed, &out, &stderr); code != 1 || !strings.Contains(stderr.String(), "another service configuration") {
		t.Fatalf("changed origin accepted: %d %s", code, &stderr)
	}
	wrongRole := append([]string{}, args...)
	wrongRole[1], wrongRole[3] = wrongRole[3], wrongRole[1]
	stderr.Reset()
	if code := runCoordinatorContext(context.Background(), "init", wrongRole, &out, &stderr); code != 1 || !strings.Contains(stderr.String(), "matching enrolled") {
		t.Fatalf("swapped roles accepted: %d %s", code, &stderr)
	}
}

func TestCoordinatorCLIInstallRejectsForeignExecutableBeforeRegistration(t *testing.T) {
	root, args, _ := coordinatorCLIFixture(t)
	t.Setenv("HOME", root)
	args = append(args, "-cluster-binary", filepath.Join(root, "not-this-executable"))
	var out, stderr bytes.Buffer
	if code := runCoordinatorContext(context.Background(), "install", args, &out, &stderr); code != 1 || !strings.Contains(stderr.String(), "must resolve to this executable") {
		t.Fatalf("install %d: %s", code, &stderr)
	}
	if _, err := os.Stat(filepath.Join(root, ".mixlab", "services")); !os.IsNotExist(err) {
		t.Fatal("rejected install published registration", err)
	}
}

func TestCoordinatorCLIHelpAndDocs(t *testing.T) {
	doc, err := os.ReadFile("../../docs/cluster-diloco.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"init", "serve", "install", "start", "stop", "status", "uninstall"} {
		var out, stderr bytes.Buffer
		if code := run([]string{"coordinator", action, "-help"}, &out, &stderr); code != 0 {
			t.Fatalf("%s help %d: %s", action, code, &stderr)
		}
		if action == "init" || action == "serve" || action == "install" {
			for _, flag := range append(clusterservice.CoordinatorPathFlags(), "coordinator-listen") {
				if !strings.Contains(stderr.String(), "-"+flag) || !strings.Contains(string(doc), "-"+flag) {
					t.Fatalf("%s missing help/docs for %s", action, flag)
				}
			}
		}
	}
	for _, args := range [][]string{{"coordinator"}, {"coordinator", "unknown"}, {"coordinator", "init", "extra"}, {"coordinator", "install", "-state-home", "/unused"}, {"coordinator", "stop", "-cluster-binary", "/unused"}} {
		var out, stderr bytes.Buffer
		if code := run(args, &out, &stderr); code != 2 {
			t.Fatalf("%v returned %d: %s", args, code, &stderr)
		}
	}
}

func fixturePrincipal(report bootstrap.Report, role trust.Role) bootstrap.PrincipalLocation {
	for _, p := range report.Principals {
		if p.Role == role {
			return p
		}
	}
	return bootstrap.PrincipalLocation{}
}

func TestDiLoCoHelpFlagsAreDocumented(t *testing.T) {
	paths, err := filepath.Glob("../../docs/*.md")
	if err != nil {
		t.Fatal(err)
	}
	var corpus strings.Builder
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		corpus.Write(b)
	}
	for _, command := range [][]string{{"coordinator", "init"}, {"coordinator", "install"}, {"coordinator", "serve"}, {"submit", "diloco"}, {"doctor"}} {
		var out, stderr bytes.Buffer
		if code := run(append(command, "-help"), &out, &stderr); code != 0 {
			t.Fatalf("help %v: %d %s", command, code, &stderr)
		}
		count := 0
		for _, line := range strings.Split(stderr.String(), "\n") {
			fields := strings.Fields(line)
			if len(fields) == 0 || !strings.HasPrefix(fields[0], "-") {
				continue
			}
			count++
			if !strings.Contains(corpus.String(), fields[0]) {
				t.Errorf("%v flag %s missing from public docs", command, fields[0])
			}
		}
		if count == 0 {
			t.Fatalf("no flags inspected for %v", command)
		}
	}
	for _, mode := range []string{"worker-initial-model", "worker-learner-probe"} {
		if !strings.Contains(corpus.String(), "`"+mode+"`") {
			t.Errorf("internal mode %s missing from docs", mode)
		}
	}
}

func TestCoordinatorRestartRefreshesStalePublicTrust(t *testing.T) {
	for _, online := range []bool{false, true} {
		name := "authority-offline"
		if online {
			name = "authority-online"
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
			root, err := filepath.EvalSymlinks(t.TempDir())
			check(err)
			check(os.Chmod(root, 0700))
			path := func(name string, kind statehome.Kind) statehome.Path {
				p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, name)}, statehome.Context{Kind: kind})
				check(err)
				return p
			}
			authorityListener, err := net.Listen("tcp", "127.0.0.1:0")
			check(err)
			defer func() { _ = authorityListener.Close() }()
			listener, err := net.Listen("tcp", "127.0.0.1:0")
			check(err)
			defer func() { _ = listener.Close() }()
			past := time.Now().Add(-20 * time.Minute)
			ids, err := bootstrap.NewIDs()
			check(err)
			ca := path("authority", statehome.Authority)
			config := bootstrap.Config{Cluster: ids.Cluster, Authority: ca, Backend: "file", Endpoint: "https://" + authorityListener.Addr().String(), Audience: "authority"}
			for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
				config.Principals = append(config.Principals, bootstrap.Target{Role: role, Principal: []string{ids.Authority, ids.Controller, ids.Coordinator}[i], Final: path(string(role)+"-principal", statehome.Principal), Staging: path(string(role)+"-stage", statehome.Enrollment)})
			}
			_, err = bootstrap.Initialize(ctx, config, past)
			check(err)
			check(clusterapp.InitializeAuthority(ctx, ca, past))
			o := clusterapp.DiLoCoCoordinatorOptions{
				Principal: config.Principals[2].Final, AdmissionPrincipal: config.Principals[1].Final,
				State: path("coordinator-state", statehome.Principal), Admission: path("admission-state", statehome.Principal), Checkpoints: path("checkpoints", statehome.Principal), Artifacts: path("artifacts", statehome.Principal),
				Endpoint: "https://" + listener.Addr().String(), Clock: func() time.Time { return past },
			}
			check(clusterapp.InitializeDiLoCoCoordinator(o))
			for _, p := range []statehome.Path{o.Principal, o.AdmissionPrincipal} {
				s, err := principal.Open(p, time.Now())
				check(err)
				_, _, activeErr := s.Active(time.Now())
				check(s.Close())
				if activeErr == nil {
					t.Fatal("fixture public trust must be stale")
				}
			}
			var attempted chan error
			if online {
				a, err := clusterapp.OpenAuthority(ctx, ca, past)
				check(err)
				defer func() { check(a.Close()) }()
				done := make(chan error, 1)
				go func() { done <- clusterapp.ServeAuthority(ctx, authorityListener, a, time.Now) }()
				defer func() { cancel(); check(<-done) }()
			} else {
				// Accept the pinned authority connection but never complete TLS.
				// Cancel only after observing the attempt, not after a wall-clock
				// race against startup's longer bounded retry policy.
				attempted = make(chan error, 1)
				probeDone := make(chan struct{})
				go func() {
					defer close(probeDone)
					conn, err := authorityListener.Accept()
					attempted <- err
					if err == nil {
						defer func() { _ = conn.Close() }()
						<-ctx.Done()
					}
				}()
				defer func() {
					cancel()
					_ = authorityListener.Close()
					<-probeDone
				}()
			}
			ready := make(chan struct{})
			o.Clock = time.Now
			o.Ready = func() { close(ready) }
			service, stop := context.WithCancel(ctx)
			defer stop()
			done := make(chan error, 1)
			go func() { done <- clusterapp.ServeInstalledDiLoCo(service, listener, o) }()
			if !online {
				select {
				case err := <-attempted:
					check(err)
				case err := <-done:
					t.Fatalf("startup exited before authority attempt: %v", err)
				case <-ctx.Done():
					stop()
					<-done
					t.Fatal("authority attempt watchdog expired")
				}
				stop()
				select {
				case err := <-done:
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("startup ignored explicit cancellation: %v", err)
					}
				case <-ctx.Done():
					t.Fatal("canceled startup did not terminate before watchdog")
				}
				select {
				case <-ready:
					t.Fatal("offline stale service reported ready")
				default:
				}
				return
			}
			select {
			case err := <-done:
				if online || err == nil {
					t.Fatalf("startup: %v", err)
				}
				select {
				case <-ready:
					t.Fatal("offline stale service reported ready")
				default:
				}
			case <-ready:
				stop()
				check(<-done)
				if !online {
					t.Fatal("stale trust accepted offline")
				}
			case <-ctx.Done():
				stop()
				<-done
				t.Fatal("startup timed out")
			}
			if online {
				for i, p := range []statehome.Path{o.Principal, o.AdmissionPrincipal} {
					s, err := principal.Open(p, time.Now())
					check(err)
					v, _, err := s.Active(time.Now())
					check(err)
					check(s.Close())
					if v.Principal != []string{ids.Coordinator, ids.Controller}[i] {
						t.Fatal("restart replaced enrolled identity")
					}
				}
			}
		})
	}
}
