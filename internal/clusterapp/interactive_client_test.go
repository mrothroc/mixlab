package clusterapp

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestInteractiveClientRealTLS(t *testing.T) {
	for _, test := range []string{"verified", "trusted-lan", "reject-root", "reject-request", "wrong-policy"} {
		t.Run(test, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			p := initialized(t)
			check(t, InitializeAuthority(ctx, p, testNow))
			a, err := OpenAuthority(ctx, p, testNow)
			check(t, err)
			defer func() { _ = a.Close() }()
			resolve := func(name string, kind statehome.Kind) statehome.Path {
				p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), name)}, statehome.Context{Kind: kind})
				check(t, err)
				return p
			}
			source, err := principal.Open(resolve("coordinator", statehome.Principal), testNow)
			check(t, err)
			defer func() { _ = source.Close() }()
			local, err := a.EnrollmentSource(ctx, source, testNow)
			check(t, err)
			l, err := net.Listen("tcp", "127.0.0.1:0")
			check(t, err)
			endpoint := "https://" + l.Addr().String()
			v, err := a.Current(ctx, testNow)
			check(t, err)
			policy := enrollment.Verified
			opts := enrollment.WindowOptions{Policy: policy, Endpoint: endpoint, Audience: "enrollment", MaxApprovals: 1}
			if test == "trusted-lan" {
				policy = enrollment.TrustedLAN
				opts.Policy = policy
				opts.CIDRs = []string{"127.0.0.0/8"}
				ifaces, err := net.Interfaces()
				check(t, err)
				for _, i := range ifaces {
					if i.Flags&net.FlagLoopback != 0 {
						opts.Interface = i.Name
						break
					}
				}
			}
			w, err := a.Enrollment.OpenWindow(ctx, opts, v, testNow)
			check(t, err)
			done := make(chan error, 1)
			go func() {
				done <- ServeEnrollment(ctx, l, local, a.Enrollment, w, a.Current, func() time.Time { return testNow })
			}()
			defer func() { cancel(); check(t, <-done) }()
			rootCalls, requestCalls := 0, 0
			ui := EnrollmentUI{
				AcceptRoot: func(ctx context.Context, p identity.Presentation) (bool, error) {
					rootCalls++
					if p.Fingerprint != a.Anchor.Fingerprint() || len(strings.Fields(p.Phrase)) != 8 {
						t.Fatal("wrong cluster presentation")
					}
					return test != "reject-root", nil
				},
				ConfirmRequest: func(ctx context.Context, p enrollment.PendingApproval) (bool, error) {
					requestCalls++
					pending, err := a.Enrollment.Pending(ctx, w.ID, v, testNow)
					check(t, err)
					if len(pending) != 1 || pending[0].ID != p.ID || pending[0].Phrase != p.Phrase || pending[0].Digest != p.Digest || len(strings.Fields(p.Phrase)) != 5 {
						t.Fatal("independent source/client presentations disagree")
					}
					if test == "reject-request" {
						return false, nil
					}
					progress, err := a.Enrollment.Decide(ctx, p.ID, p.Digest, true, v, testNow)
					check(t, err)
					if progress.Result != nil {
						t.Fatal("issued before client comparison")
					}
					return true, nil
				},
			}
			if test == "wrong-policy" {
				policy = enrollment.TrustedLAN
			}
			stage, final := resolve("stage", statehome.Enrollment), resolve("node", statehome.Principal)
			state, err := EnrollInteractive(ctx, InteractiveOptions{Endpoint: endpoint, Policy: policy, Purpose: enrollment.NodeEnrollment, Backend: "file", Stage: stage, Final: final, UI: ui}, func() time.Time { return testNow })
			if test == "verified" || test == "trusted-lan" {
				check(t, err)
				if state.Role != trust.Node || state.Fingerprint != a.Anchor.Fingerprint() || state.EnvelopeKey == nil {
					t.Fatal("invalid installed identity")
				}
				if test == "verified" && (rootCalls != 1 || requestCalls != 1) {
					t.Fatal("missing comparisons")
				}
			} else {
				if err == nil {
					t.Fatal("accepted rejected enrollment")
				}
				if _, err := os.Lstat(final.Dir()); !os.IsNotExist(err) {
					t.Fatal("published rejected principal", err)
				}
			}
			if (test == "trusted-lan" || test == "wrong-policy") && (rootCalls != 0 || requestCalls != 0) {
				t.Fatal("policy silently changed")
			}
			if test == "reject-root" && (rootCalls != 1 || requestCalls != 0) {
				t.Fatal("requested certificate before root approval")
			}
			if _, err := os.Lstat(stage.Dir()); !os.IsNotExist(err) {
				t.Fatal("provisional staging not cleaned", err)
			}
		})
	}
}
