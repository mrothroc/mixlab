package enrollment

import (
	"bytes"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"fmt"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestPrincipalRenewalExactRetry(t *testing.T) {
	for _, role := range []trust.Role{trust.Node, trust.Controller, trust.Coordinator} {
		t.Run(string(role), func(t *testing.T) {
			f := setup(t)
			k := key(t)
			pid := id(t)
			var leaf []byte
			var err error
			if role == trust.Node {
				x, e := ecdh.X25519().GenerateKey(rand.Reader)
				check(t, e)
				leaf, err = certificates.IssueNode(f.a.Anchor, f.a.Issuer, f.a.IssuerKey, pid, k.Public(), x.PublicKey().Bytes(), now)
			} else {
				leaf, err = certificates.Issue(f.a.Anchor, f.a.Issuer, f.a.IssuerKey, certificates.Principal, role, pid, k.Public(), now)
			}
			check(t, err)
			f.issuer.calls = 0
			chain := [][]byte{leaf, f.a.Issuer, f.a.Anchor.DER()}
			q, err := NewRenewalRequest(chain, k)
			check(t, err)
			out, err := f.s.Renew(ctx, chain, q, f.v, now)
			check(t, err)
			if f.issuer.calls != 1 {
				t.Fatal("issuance count", f.issuer.calls)
			}
			if bytes.Equal(leaf, out.Chain[0]) {
				t.Fatal("did not issue renewed certificate")
			}
			reopened, err := Open(f.p, f.a, f.v, now)
			check(t, err)
			again, err := reopened.Renew(ctx, chain, q, f.v, now)
			check(t, err)
			if !bytes.Equal(again.Chain[0], out.Chain[0]) || f.issuer.calls != 1 {
				t.Fatal("retry reissued")
			}
			other, err := NewRenewalRequest(chain, k)
			check(t, err)
			if _, err := f.s.Renew(ctx, chain, other, f.v, now); err == nil {
				t.Fatal("multiple successors issued")
			}
		})
	}
}

func TestRenewalJournalFailureRecovery(t *testing.T) {
	for _, stage := range []string{"pending", "issued"} {
		for _, after := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/after=%v", stage, after), func(t *testing.T) {
				f := setup(t)
				k := key(t)
				leaf, err := certificates.Issue(f.a.Anchor, f.a.Issuer, f.a.IssuerKey, certificates.Principal, trust.Controller, id(t), k.Public(), now)
				check(t, err)
				chain := [][]byte{leaf, f.a.Issuer, f.a.Anchor.DER()}
				q, err := NewRenewalRequest(chain, k)
				check(t, err)
				f.issuer.calls = 0
				f.s.store = &failingJournal{journal: f.p, stage: stage, after: after}
				out, err := f.s.Renew(ctx, chain, q, f.v, now)
				if err == nil || len(out.Chain) != 0 {
					t.Fatal("uncertain renewal escaped")
				}
				if stage == "pending" && f.issuer.calls != 0 {
					t.Fatal("issuer called before durable renewal intent")
				}
				s, err := Open(f.p, f.a, f.v, now)
				check(t, err)
				r, err := s.Renew(ctx, chain, q, f.v, now)
				check(t, err)
				if len(r.Chain) != 3 {
					t.Fatal("renewal not recovered")
				}
				if stage == "issued" && after && f.issuer.calls != 1 {
					t.Fatal("reissued committed renewal")
				}
			})
		}
	}
}

func TestRenewalRejectsExpiredRevokedAndSubstitutedIdentity(t *testing.T) {
	for _, mode := range []string{"expired", "revoked", "wrong-proof", "wrong-chain"} {
		t.Run(mode, func(t *testing.T) {
			f := setup(t)
			k := key(t)
			pid := id(t)
			leaf, err := certificates.Issue(f.a.Anchor, f.a.Issuer, f.a.IssuerKey, certificates.Principal, trust.Controller, pid, k.Public(), now)
			check(t, err)
			if mode == "expired" {
				c, err := x509.ParseCertificate(leaf)
				check(t, err)
				parent, err := x509.ParseCertificate(f.a.Issuer)
				check(t, err)
				c.NotAfter = now.Add(time.Second)
				leaf, err = x509.CreateCertificate(rand.Reader, c, parent, k.Public(), f.a.IssuerKey)
				check(t, err)
			}
			chain := [][]byte{leaf, f.a.Issuer, f.a.Anchor.DER()}
			q, err := NewRenewalRequest(chain, k)
			check(t, err)
			v := f.v
			at := now
			switch mode {
			case "expired":
				at = now.Add(2 * time.Second)
			case "revoked":
				v = f.view(t, 2, now, []trust.Revocation{{Kind: "principal", ID: pid, Mode: "compromise", Reason: "test", FirstGeneration: 2}})
			case "wrong-proof":
				q.Signature = ed25519.Sign(key(t), []byte("wrong request"))
			case "wrong-chain":
				chain = f.a.PrincipalChain
			}
			before := f.issuer.calls
			if _, err := f.s.Renew(ctx, chain, q, v, at); err == nil {
				t.Fatal("invalid renewal accepted")
			}
			if f.issuer.calls != before {
				t.Fatal("invalid renewal reached issuer")
			}
		})
	}
}
