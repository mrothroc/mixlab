package principal_test

import (
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/principal"
)

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func key(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	_, k, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	return k
}
func id(t *testing.T) string { t.Helper(); v, err := certificates.NewID(); check(t, err); return v }

type fixture struct {
	stage, final statehome.Path
	a            trust.Anchor
	keys         *securekeys.Store
	state        principal.State
	ik, sk       crypto.Signer
	sc           []byte
	now          time.Time
}

func setup(t *testing.T, role trust.Role) fixture {
	t.Helper()
	f := fixture{now: time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)}
	d, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(d, 0700))
	f.stage, err = statehome.Resolve(statehome.Options{ExactDir: filepath.Join(d, "stage")}, statehome.Context{Kind: statehome.Enrollment})
	check(t, err)
	f.final, err = statehome.Resolve(statehome.Options{ExactDir: filepath.Join(d, "final")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	check(t, f.stage.Publish(func(statehome.Path) error { return nil }))
	rk := key(t)
	f.ik = key(t)
	f.sk = key(t)
	root, err := certificates.CreateRoot(id(t), rk, f.now)
	check(t, err)
	fp, err := trust.RootFingerprint(rk.Public())
	check(t, err)
	f.a, err = trust.PinRoot(root, fp, f.now)
	check(t, err)
	issuer, err := certificates.Issue(f.a, root, rk, certificates.Issuer, "", "", f.ik.Public(), f.now)
	check(t, err)
	f.sc, err = certificates.Issue(f.a, root, rk, certificates.SnapshotSigner, "", "", f.sk.Public(), f.now)
	check(t, err)
	f.keys, err = securekeys.OpenFile(f.stage, fp)
	check(t, err)
	t.Cleanup(func() { _ = f.keys.Close() })
	m, err := keylifecycle.Initialize(context.Background(), f.stage, f.keys, string(role))
	check(t, err)
	h, err := m.Create(context.Background(), keylifecycle.Principal)
	check(t, err)
	f.state = principal.State{Version: principal.Version, Cluster: f.a.Cluster(), Role: role, Principal: id(t), Key: *h.Active, Root: root, Fingerprint: fp}
	var leaf []byte
	if role == trust.Node {
		e, err := m.Create(context.Background(), keylifecycle.NodeEnvelope)
		check(t, err)
		f.state.EnvelopeKey = e.Active
		leaf, err = certificates.IssueNode(f.a, issuer, f.ik, f.state.Principal, ed25519.PublicKey(h.Active.PublicKey), e.Active.PublicKey, f.now)
		check(t, err)
	} else {
		leaf, err = certificates.Issue(f.a, issuer, f.ik, certificates.Principal, role, f.state.Principal, ed25519.PublicKey(h.Active.PublicKey), f.now)
		check(t, err)
	}
	f.state.Chain = [][]byte{leaf, issuer, root}
	e, err := trust.SignAuthorityEndpoints(f.a, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: f.a.Cluster(), Audience: "authority", URLs: []string{"https://authority.example"}, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(90 * 24 * time.Hour).Unix()}, rk, f.now)
	check(t, err)
	f.state.Snapshot, err = trust.SignSnapshot(f.a, trust.Snapshot{Version: trust.SnapshotVersion, Cluster: f.a.Cluster(), Generation: 1, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(trust.SnapshotLifetime).Unix(), Issuers: [][]byte{issuer}, EligibleRoles: []trust.Role{role}, Endpoints: e}, f.sc, f.sk, f.now)
	check(t, err)
	return f
}
func (f fixture) snapshot(t *testing.T, at time.Time, revoked bool) trust.SignedSnapshot {
	t.Helper()
	p := f.state.Snapshot.Payload
	p.Generation++
	p.IssuedAt = at.Unix()
	p.ExpiresAt = at.Add(trust.SnapshotLifetime).Unix()
	if revoked {
		p.Revocations = []trust.Revocation{{Kind: "principal", ID: f.state.Principal, Mode: "compromise", Reason: "test", FirstGeneration: 2}}
	}
	s, err := trust.SignSnapshot(f.a, p, f.sc, f.sk, at)
	check(t, err)
	return s
}
func (f fixture) installed(t *testing.T) *principal.Store {
	t.Helper()
	check(t, principal.Install(context.Background(), f.stage, f.a, f.state, f.now))
	check(t, f.final.Promote(f.stage))
	s, err := principal.Open(f.final, f.now)
	check(t, err)
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestPrincipalInstallAndRefresh(t *testing.T) {
	for _, role := range []trust.Role{trust.Node, trust.Controller, trust.Coordinator, trust.Authority} {
		t.Run(string(role), func(t *testing.T) {
			f := setup(t, role)
			s := f.installed(t)
			r, k, err := s.Active(f.now)
			check(t, err)
			if r.Principal != f.state.Principal || k == nil {
				t.Fatal("identity changed")
			}
			stale := f.now.Add(trust.SnapshotLifetime + time.Second)
			if _, _, err := s.Active(stale); err == nil {
				t.Fatal("stale authorization")
			}
			if _, err := s.View(stale); err != nil {
				t.Fatal("stale history cannot be inspected", err)
			}
			check(t, s.Refresh(context.Background(), f.snapshot(t, stale, false), stale))
			_, _, err = s.Active(stale)
			check(t, err)
			if err := s.Refresh(context.Background(), f.state.Snapshot, stale); err == nil {
				t.Fatal("rollback accepted")
			}
		})
	}
}

func TestPrincipalInstallRejectsWrongPinAndKeys(t *testing.T) {
	f := setup(t, trust.Node)
	other := setup(t, trust.Node)
	if err := principal.Install(context.Background(), f.stage, other.a, f.state, f.now); err == nil {
		t.Fatal("wrong pin installed")
	}
	bad := f.state
	bad.EnvelopeKey = nil
	if err := principal.Install(context.Background(), f.stage, f.a, bad, f.now); err == nil {
		t.Fatal("missing node envelope installed")
	}
	check(t, principal.Install(context.Background(), f.stage, f.a, f.state, f.now))
	if err := principal.Install(context.Background(), f.stage, f.a, f.state, f.now); err == nil {
		t.Fatal("overwrote installed state")
	}
}

func TestPrincipalRenewalEligibility(t *testing.T) {
	for _, mode := range []string{"valid", "wrong-key", "wrong-id", "revoked", "expired"} {
		t.Run(mode, func(t *testing.T) {
			f := setup(t, trust.Controller)
			s := f.installed(t)
			at := f.now.Add(time.Hour)
			if mode == "expired" {
				at = f.now.Add(31 * 24 * time.Hour)
			}
			pub := ed25519.PublicKey(f.state.Key.PublicKey)
			pid := f.state.Principal
			if mode == "wrong-key" {
				pub = key(t).Public().(ed25519.PublicKey)
			}
			if mode == "wrong-id" {
				pid = id(t)
			}
			leaf, err := certificates.Issue(f.a, f.state.Chain[1], f.ik, certificates.Principal, trust.Controller, pid, pub, at)
			check(t, err)
			chain := [][]byte{leaf, f.state.Chain[1], f.a.DER()}
			err = s.Renew(context.Background(), chain, f.snapshot(t, at, mode == "revoked"), at)
			if mode == "valid" {
				check(t, err)
				_, _, err = s.Active(at)
				check(t, err)
			} else if err == nil {
				t.Fatal("invalid renewal accepted")
			}
		})
	}
}
