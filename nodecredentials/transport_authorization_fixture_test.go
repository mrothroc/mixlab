package nodecredentials

import (
	"context"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/authority"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

type transportAuthorizationState struct {
	credential *Transport
	path       statehome.Path
	anchor     trust.Anchor
	snapshots  *authority.Store
	view       trust.VerifiedSnapshot
	now        time.Time
	chain      [][]byte
	key        ed25519.PrivateKey
}

func (f *transportAuthorizationState) sign(view trust.VerifiedSnapshot, now time.Time) NodeSigner {
	return func(_ context.Context, q trust.SignRequest) (trust.SignedProof, error) {
		return trust.SignPrincipalProof(f.anchor, f.chain, f.key, view, q, now)
	}
}

func transportAuthorizationFixture(t *testing.T) *transportAuthorizationState {
	t.Helper()
	ctx := context.Background()
	now := time.Date(2026, 10, 17, 0, 0, 0, 0, time.UTC)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	credentialCheck(t, err)
	resolve := func(name string, kind statehome.Kind) statehome.Path {
		t.Helper()
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, name)}, statehome.Context{Kind: kind})
		credentialCheck(t, err)
		return p
	}
	ids, err := bootstrap.NewIDs()
	credentialCheck(t, err)
	root := resolve("authority-state", statehome.Authority)
	cfg := bootstrap.Config{Cluster: ids.Cluster, Authority: root, Backend: "file", Endpoint: "https://authority.example:7443", Audience: "authority"}
	for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
		cfg.Principals = append(cfg.Principals, bootstrap.Target{Role: role, Principal: []string{ids.Authority, ids.Controller, ids.Coordinator}[i], Final: resolve(string(role), statehome.Principal), Staging: resolve(string(role)+"-stage", statehome.Enrollment)})
	}
	_, err = bootstrap.Initialize(ctx, cfg, now)
	credentialCheck(t, err)
	m, err := bootstrap.LoadAuthority(root, now)
	credentialCheck(t, err)
	keys, err := securekeys.OpenSelected("file", root, m.IssuerKey.Scope)
	credentialCheck(t, err)
	t.Cleanup(func() { credentialCheck(t, keys.Close()) })
	issuer, err := keys.Signer(m.IssuerKey)
	credentialCheck(t, err)
	snapshotKey, err := keys.Signer(m.SnapshotKey)
	credentialCheck(t, err)
	snapshots, err := authority.Initialize(ctx, root, m.Anchor, m.SnapshotSigner, snapshotKey, m.InitialSnapshot, now)
	credentialCheck(t, err)
	view, err := trust.VerifySnapshot(m.Anchor, m.InitialSnapshot, now)
	credentialCheck(t, err)
	identity, err := principal.Open(m.AuthorityPrincipal, now)
	credentialCheck(t, err)
	t.Cleanup(func() { credentialCheck(t, identity.Close()) })
	state, signer, err := identity.Active(now)
	credentialCheck(t, err)
	enroll, err := enrollment.Initialize(ctx, root, enrollment.Authority{Anchor: m.Anchor, Issuer: m.Issuer, IssuerKey: issuer, PrincipalChain: state.Chain, PrincipalKey: signer}, view, now)
	credentialCheck(t, err)
	invite, err := enroll.Invite(ctx, enrollment.NodeEnrollment, cfg.Endpoint, cfg.Audience, time.Minute, view, now)
	credentialCheck(t, err)
	defer invite.Clear()
	_, key, err := ed25519.GenerateKey(rand.Reader)
	credentialCheck(t, err)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	credentialCheck(t, err)
	request, err := enrollment.NewRequest(invite, key, envelope.PublicKey().Bytes(), now)
	credentialCheck(t, err)
	node, err := enroll.Consume(ctx, request, invite.Secret, view, now)
	credentialCheck(t, err)
	id := func(value string) string { return strings.Repeat(value, 32) }
	scope := workload.Scope{Profile: workload.LearnerVersion, Cluster: ids.Cluster, Controller: ids.Controller, Node: node.Approval.Principal, Lease: id("1"), Run: id("2"), Job: id("3"), Attempt: id("4"), Workload: id("5"), Generation: 1, ManifestHash: strings.Repeat("a", 64), Created: now.Unix(), AdmitUntil: now.Add(30 * time.Minute).Unix(), Deadline: now.Add(time.Hour).Unix(), Worker: id("6"), PlanHash: strings.Repeat("b", 64), Coordinator: ids.Coordinator}
	path := resolve("credential", statehome.Agent)
	c, err := InitializeTransport(ctx, path, m.Anchor, scope, "file")
	credentialCheck(t, err)
	f := &transportAuthorizationState{credential: c, path: path, anchor: m.Anchor, snapshots: snapshots, view: view, now: now, chain: node.Chain, key: key}
	t.Cleanup(func() { credentialCheck(t, f.credential.Close()) })
	return f
}
