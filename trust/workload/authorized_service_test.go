package workload

import (
	"bytes"
	"context"
	"crypto/ecdh"
	"crypto/rand"
	"encoding/json"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func TestAuthorizedWorkloadRevocationBlocksAcceptanceAndReplay(t *testing.T) {
	for _, committed := range []bool{false, true} {
		for _, who := range []string{"node", "controller"} {
			for _, mode := range []string{"prospective", "compromise"} {
				name := who + "/" + mode
				if committed {
					name += "/committed"
				}
				t.Run(name, func(t *testing.T) {
					f := newFixture(t)
					auth := f.authorize(t, f.grant)
					if committed {
						_, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
						check(t, err)
					}
					id := f.grant.Request.Request.Scope.Node
					if who == "controller" {
						id = f.grant.Request.Request.Scope.Controller
					}
					f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: id, Mode: mode, Reason: "test", FirstGeneration: 2}}
					f.advance(t, time.Second)
					before := f.issuer.calls
					if _, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now); err == nil || f.issuer.calls != before {
						t.Fatal("revoked principal reached issuance or delivery")
					}
				})
			}
		}
	}
}

func (f *fixture) renewAuthorizationSigner(t *testing.T, who string) string {
	t.Helper()
	chain := f.node
	if who == "controller" {
		chain = f.controller
	}
	_, old, err := certificates.Parse(chain[0])
	check(t, err)
	key := newKey(t)
	var der []byte
	if who == "node" {
		envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
		check(t, err)
		der, err = certificates.IssueNode(f.a.Anchor, f.a.Issuer, f.issuer, old.Principal, key.Public(), envelope.PublicKey().Bytes(), f.now)
		check(t, err)
		f.node, f.nodeKey = [][]byte{der, f.a.Issuer, f.a.Anchor.DER()}, key
	} else {
		der, err = certificates.Issue(f.a.Anchor, f.a.Issuer, f.issuer, certificates.Principal, trust.Controller, old.Principal, key.Public(), f.now)
		check(t, err)
		f.controller, f.controllerKey = [][]byte{der, f.a.Issuer, f.a.Anchor.DER()}, key
	}
	return old.Serial
}

func TestAuthorizedWorkloadReplacementCannotEraseCompromisedHistory(t *testing.T) {
	for _, who := range []string{"node", "controller"} {
		for _, mode := range []string{"prospective", "compromise"} {
			t.Run(who+"/"+mode, func(t *testing.T) {
				f := newFixture(t)
				auth := f.authorize(t, f.grant)
				out, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
				check(t, err)
				f.advance(t, time.Minute)
				serial := f.renewAuthorizationSigner(t, who)
				f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "certificate", ID: serial, Mode: mode, Reason: "test", FirstGeneration: f.view.Generation()}}
				f.refresh(t)
				replacement := f.authorize(t, f.grant)
				check(t, VerifyAuthorizedGrant(f.a.Anchor, f.view, replacement, f.now))
				before := f.issuer.calls
				again, err := f.s.IssueAuthorized(context.Background(), f.controller, replacement, f.view, f.now)
				if mode == "compromise" {
					if err == nil {
						t.Fatal("fresh supplements erased compromised acceptance")
					}
				} else {
					check(t, err)
					if !bytes.Equal(out.Chain[0], again.Chain[0]) {
						t.Fatal("renewal changed certificate")
					}
				}
				if f.issuer.calls != before {
					t.Fatal("retry reached issuer")
				}
			})
		}
	}
}

func TestAuthorizedWorkloadPublicationRecovery(t *testing.T) {
	for _, after := range []bool{false, true} {
		t.Run(map[bool]string{false: "before", true: "after"}[after], func(t *testing.T) {
			f := newFixture(t)
			f.advance(t, 20*time.Minute)
			auth := f.authorize(t, f.grant)
			f.s.store = &failingPublication{journal: f.path, after: after}
			out, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
			if err == nil || len(out.Chain) != 0 {
				t.Fatal("uncertain publication exposed result")
			}
			issuedAt := f.now.Unix()
			f.advance(t, 20*time.Minute)
			f.s, err = Open(f.path, f.a, f.view, f.now)
			check(t, err)
			out, err = f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
			check(t, err)
			check(t, ValidateResult(f.a.Anchor, f.grant, out, f.now))
			want := 2
			if after {
				want = 1
			}
			if f.issuer.calls != want || out.Binding.IssuedAt != issuedAt || out.GrantHash != digest(f.grant) {
				t.Fatal("recovery changed immutable issuance")
			}
		})
	}
}

func TestAuthorizedWorkloadRejectsSubstitutionAndBadReplacement(t *testing.T) {
	f := newFixture(t)
	auth := f.authorize(t, f.grant)
	_, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
	check(t, err)
	f.advance(t, time.Minute)
	changed := f.sign(t, f.grant.Request.Request.Scope, newKey(t))
	validButDifferent := f.authorize(t, changed)
	check(t, VerifyAuthorizedGrant(f.a.Anchor, f.view, validButDifferent, f.now))
	if _, err := f.s.IssueAuthorized(context.Background(), f.controller, validButDifferent, f.view, f.now); err == nil {
		t.Fatal("replaced committed grant with a fresh key")
	}
	for _, who := range []string{"node", "controller"} {
		t.Run(who, func(t *testing.T) {
			replacement := f.authorize(t, f.grant)
			if who == "node" {
				replacement.NodeAuthorization.Proof.Signature[0] ^= 1
			} else {
				replacement.ControllerAuthorization.Signature[0] ^= 1
			}
			if _, err := f.s.IssueAuthorized(context.Background(), f.controller, replacement, f.view, f.now); err == nil {
				t.Fatal("ignored bad replacement signature")
			}
		})
	}
	if f.issuer.calls != 1 {
		t.Fatal("replacement reissued")
	}
}

func TestAuthorizedWorkloadRejectsFakeOriginalReceipts(t *testing.T) {
	f := newFixture(t)
	auth := f.authorize(t, f.grant)
	_, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
	check(t, err)
	old, r, err := f.s.load()
	check(t, err)
	r.Entries[0].NodeCommit.Digest = digest(f.grant.Request.Proof)
	r.Entries[0].GrantCommit.Digest = digest(f.grant.Proof)
	b, err := json.Marshal(r)
	check(t, err)
	check(t, f.path.CompareAndSwap(stateFile, old, b))
	if _, err := Open(f.path, f.a, f.view, f.now); err == nil {
		t.Fatal("supplemental entry accepted fake original-proof receipts")
	}
}

func TestAuthorizedWorkloadCannotExtendAdmission(t *testing.T) {
	f := newFixture(t)
	auth := f.authorize(t, f.grant)
	_, err := f.s.IssueAuthorized(context.Background(), f.controller, auth, f.view, f.now)
	check(t, err)
	f.advance(t, time.Hour)
	// Cryptographic signing alone cannot extend the payload's admission time.
	replacement := f.authorize(t, f.grant)
	if VerifyAuthorizedGrant(f.a.Anchor, f.view, replacement, f.now) == nil {
		t.Fatal("expired authorization verified")
	}
	if _, err := f.s.IssueAuthorized(context.Background(), f.controller, replacement, f.view, f.now); err == nil || f.issuer.calls != 1 {
		t.Fatal("fresh supplement revived expired admission")
	}
}
