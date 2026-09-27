package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/principal"
)

func availableAddress(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	return addr
}

func TestEnrollmentCLIJourneys(t *testing.T) {
	for _, scenario := range []string{"provisioned", "verified", "trusted-lan", "verified-mdns", "trusted-lan-mdns"} {
		t.Run(scenario, func(t *testing.T) {
			policy := strings.TrimSuffix(scenario, "-mdns")
			browse := strings.HasSuffix(scenario, "-mdns")
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			dir, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(dir, 0700); err != nil {
				t.Fatal(err)
			}
			address := availableAddress(t)
			var initOut, initErr bytes.Buffer
			coordinator := filepath.Join(dir, "coordinator")
			if code := runInit([]string{"-state-home", filepath.Join(dir, "state"), "-key-backend", "file", "-trust-listen", address, "-coordinator-principal-state-dir", coordinator}, &initOut, &initErr); code != 0 {
				t.Fatalf("init %d: %s", code, &initErr)
			}
			var report bootstrap.Report
			if err := json.Unmarshal(initOut.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			read, write := io.Pipe()
			defer func() { _ = read.Close(); _ = write.Close() }()
			done := make(chan int, 1)
			var sourceErrors bytes.Buffer
			invite := filepath.Join(dir, "invitation.json")
			clientArgs := []string{"-enrollment-policy", policy, "-principal-state-dir", filepath.Join(dir, "node"), "-key-backend", "file"}
			if policy == "provisioned" {
				clientArgs = append(clientArgs, "-enrollment-provisioning-file", invite)
				go func() {
					done <- runInviteContext(ctx, []string{"-cluster-state-dir", report.AuthorityDir, "-invite-output", invite}, write, &sourceErrors)
					_ = write.Close()
				}()
			} else {
				address = availableAddress(t)
				if browse {
					clientArgs = append(clientArgs, "-enrollment-discover", "mdns")
				} else {
					clientArgs = append(clientArgs, "-enrollment-coordinator", "https://"+address)
				}
				args := []string{"-cluster-state-dir", report.AuthorityDir, "-principal-state-dir", coordinator, "-enrollment-listen", address, "-enrollment-policy", policy}
				if policy == "trusted-lan" {
					ifaces, err := net.Interfaces()
					if err != nil {
						t.Fatal(err)
					}
					for _, i := range ifaces {
						if i.Flags&net.FlagLoopback != 0 {
							args = append(args, "-enrollment-interface", i.Name, "-enrollment-cidr", "127.0.0.0/8")
							break
						}
					}
				}
				go func() {
					done <- runEnrollmentServeContext(ctx, args, write, &sourceErrors, func(context.Context, enrollment.PendingApproval) (bool, error) { return true, nil })
					_ = write.Close()
				}()
			}
			// Reporting occurs only after the listener and protected window/file exist.
			line, err := bufio.NewReader(read).ReadString('\n')
			if err != nil {
				cancel()
				<-done
				t.Fatalf("source readiness: %v: %s", err, &sourceErrors)
			}
			if !json.Valid([]byte(line)) {
				t.Fatalf("invalid source report %q", line)
			}
			ui := clusterapp.EnrollmentUI{AcceptRoot: func(context.Context, identity.Presentation) (bool, error) { return true, nil }, ConfirmRequest: func(context.Context, enrollment.PendingApproval) (bool, error) { return true, nil }}
			var out, errs bytes.Buffer
			// Deliberately false TXT claims must not select the root or policy.
			provider := hintProvider{hints: []discovery.Hint{{Service: discovery.Enrollment, Endpoint: address, Claims: discovery.Claims{RootFingerprint: "not-the-server-root", Policies: []string{"unrelated-policy"}}}}}
			code := runEnrollWithDiscovery(ctx, clientArgs, &out, &errs, ui, provider)
			cancel()
			sourceCode := <-done
			if code != 0 {
				t.Fatalf("enroll %d: %s; source: %s", code, &errs, &sourceErrors)
			}
			if sourceCode != 0 {
				t.Fatalf("source %d: %s", sourceCode, &sourceErrors)
			}
			p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, "node")}, statehome.Context{Kind: statehome.Principal})
			if err != nil {
				t.Fatal(err)
			}
			s, err := principal.Open(p, time.Now())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = s.Close() }()
			if _, _, err := s.Active(time.Now()); err != nil {
				t.Fatal(err)
			}
			if policy == "provisioned" {
				if _, err := os.Stat(invite); !os.IsNotExist(err) {
					t.Fatal("consumed provisioning file not deleted", err)
				}
			}
			staging, err := filepath.Glob(filepath.Join(dir, ".enrollment-*"))
			if err != nil || len(staging) != 0 {
				t.Fatal("staging leak", staging, err)
			}
		})
	}
}

func TestEnrollmentCLIValidationAndHelp(t *testing.T) {
	for _, args := range [][]string{
		{"invite"}, {"invite", "-invite-output", "/tmp/file", "-invite-ttl", "2h"}, {"invite", "-invite-output", "/tmp/file", "-invite-purpose", "external-worker-bootstrap"},
		{"enroll", "-enrollment-policy", "auto", "-principal-state-dir", "/tmp/never"},
		{"enroll", "-enrollment-policy", "verified", "-enrollment-provisioning-file", "/tmp/no", "-principal-state-dir", "/tmp/never"},
		{"enroll", "-enrollment-policy", "trusted-lan", "-enrollment-coordinator", "https://example:443", "-enrollment-purpose", "controller-enrollment", "-principal-state-dir", "/tmp/never"},
		{"enrollment", "serve"}, {"authority", "serve", "-trust-advertise", "invalid"}, {"revoke"},
	} {
		var out, errs bytes.Buffer
		if code := run(args, &out, &errs); code != 2 || out.Len() != 0 || errs.Len() == 0 {
			t.Fatalf("%v: %d out=%s err=%s", args, code, &out, &errs)
		}
	}
	for _, command := range [][]string{{"invite"}, {"enroll"}, {"enrollment", "serve"}, {"authority", "serve"}, {"revoke"}} {
		var out, errs bytes.Buffer
		if code := run(append(command, "-help"), &out, &errs); code != 0 || !strings.Contains(errs.String(), "Usage:") {
			t.Fatalf("help %v: %d %s", command, code, &errs)
		}
	}
}
