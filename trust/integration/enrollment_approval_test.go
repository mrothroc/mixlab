package integration

import (
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

func enrollmentPath(t *testing.T, kind statehome.Kind) statehome.Path {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: kind})
	check(t, err)
	return p
}

func TestEnrollmentTLSRealChannelApproval(t *testing.T) {
	for _, policy := range []enrollment.Policy{enrollment.Verified, enrollment.TrustedLAN} {
		t.Run(string(policy), func(t *testing.T) {
			f := newTLSFixture(t)
			f.snapshot.Payload.EligibleRoles = append(f.snapshot.Payload.EligibleRoles, trust.Node)
			f.signSnapshot(t)
			authority := enrollment.Authority{Anchor: f.a, PrincipalChain: f.server.Chain, PrincipalKey: f.server.Key, Issuer: f.issuer, IssuerKey: f.issuerKey}
			svc, err := enrollment.Initialize(context.Background(), enrollmentPath(t, statehome.Authority), authority, f.view, f.now)
			check(t, err)
			sourceIdentity := f.leaf(t, trust.Coordinator, id(t))
			scfg, err := enrollmenttls.ServerConfig(sourceIdentity, f.clock)
			check(t, err)
			ccfg, err := enrollmenttls.ClientConfig(func(chain [][]byte, now time.Time) error {
				_, err := f.verify(trust.Coordinator, principalID(t, f, sourceIdentity.Chain))(chain, now)
				return err
			}, f.clock)
			check(t, err)
			server, client, err := tlsPair(t, scfg, ccfg)
			check(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			source, err := enrollmenttls.New(ctx, server)
			check(t, err)
			remote, err := enrollmenttls.New(ctx, client)
			check(t, err)
			t.Cleanup(func() { _ = source.Close(); _ = remote.Close() })
			n, err := source.Network()
			check(t, err)
			o := enrollment.WindowOptions{Policy: policy, Endpoint: "https://enroll.example:7444", Audience: "source", MaxApprovals: 1}
			if policy == enrollment.TrustedLAN {
				o.Interface = n.Interface
				o.CIDRs = []string{"127.0.0.0/8"}
			}
			w, err := svc.OpenWindow(ctx, o, f.view, f.now)
			check(t, err)
			store, err := securekeys.OpenFile(enrollmentPath(t, statehome.Enrollment), f.a.Fingerprint())
			check(t, err)
			defer func() { _ = store.Close() }()
			h, err := store.Generate()
			check(t, err)
			signer, err := store.Signer(h)
			check(t, err)
			x, err := ecdh.X25519().GenerateKey(rand.Reader)
			check(t, err)
			q, err := enrollment.NewInteractiveRequest(f.a, w, enrollment.NodeEnrollment, signer, x.PublicKey().Bytes(), f.now)
			check(t, err)
			p, err := svc.Begin(ctx, q, source, f.view, f.now)
			check(t, err)
			d, err := p.Pairing.Digest()
			check(t, err)
			exporter, err := remote.Bind(d)
			check(t, err)
			_, digest, err := enrollment.PairingPresentation(p.Pairing, exporter)
			clear(exporter)
			check(t, err)
			if digest != p.PairingDigest {
				t.Fatal("real TLS phrase mismatch")
			}
			if policy == enrollment.Verified {
				p, err = svc.ConfirmClient(ctx, p.ID, digest, source, f.view, f.now)
				check(t, err)
				if p.Result != nil {
					t.Fatal("issued before local administrator approval")
				}
				p, err = svc.Decide(ctx, p.ID, digest, true, f.view, f.now)
				check(t, err)
			}
			if p.Result == nil {
				t.Fatal("no enrollment result")
			}
			_, err = enrollment.ValidateInteractiveResult(f.a, w, q, p.Pairing, digest, *p.Result, f.now)
			check(t, err)
			check(t, svc.Disconnect(ctx, source))
			if _, err := svc.Poll(ctx, p.ID, source, f.view, f.now); err == nil {
				t.Fatal("terminated connection reused")
			}
		})
	}
}

func principalID(t *testing.T, f *tlsFixture, chain [][]byte) string {
	t.Helper()
	p, err := trust.AuthenticatePrincipal(f.a, f.view, chain, f.now)
	check(t, err)
	return p.Principal
}
