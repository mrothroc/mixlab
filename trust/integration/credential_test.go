package integration

import (
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

func must[T any](t *testing.T, v T, err error) T {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return v
}
func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func key(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	_, k, e := ed25519.GenerateKey(rand.Reader)
	return must(t, k, e)
}
func id(t *testing.T) string { t.Helper(); v, e := certificates.NewID(); return must(t, v, e) }

func TestSignedEnvelopeWithDurableReplay(t *testing.T) {
	now := time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)
	rk, ik, sk, ck, nk := key(t), key(t), key(t), key(t), key(t)
	root, err := certificates.CreateRoot(id(t), rk, now)
	check(t, err)
	fp, err := trust.RootFingerprint(rk.Public())
	check(t, err)
	a, err := trust.PinRoot(root, fp, now)
	check(t, err)
	issuer, err := certificates.Issue(a, root, rk, certificates.Issuer, "", "", ik.Public(), now)
	check(t, err)
	signer, err := certificates.Issue(a, root, rk, certificates.SnapshotSigner, "", "", sk.Public(), now)
	check(t, err)
	cid, nid := id(t), id(t)
	controller, err := certificates.Issue(a, issuer, ik, certificates.Principal, trust.Controller, cid, ck.Public(), now)
	check(t, err)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	store, err := securekeys.OpenFile(p, fp)
	check(t, err)
	h, err := store.GenerateEnvelope()
	check(t, err)
	t.Cleanup(func() { check(t, store.Delete(h)); check(t, store.Close()) })
	node, err := certificates.IssueNode(a, issuer, ik, nid, nk.Public(), h.PublicKey, now)
	check(t, err)
	eps, err := trust.SignAuthorityEndpoints(a, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: a.Cluster(), Audience: "authority", URLs: []string{"https://authority.example"}, IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Hour).Unix()}, rk, now)
	check(t, err)
	snapshot, err := trust.SignSnapshot(a, trust.Snapshot{Version: trust.SnapshotVersion, Cluster: a.Cluster(), Generation: 1, IssuedAt: now.Unix(), ExpiresAt: now.Add(trust.SnapshotLifetime).Unix(), Issuers: [][]byte{issuer}, EligibleRoles: []trust.Role{trust.Controller, trust.Node}, Endpoints: eps}, signer, sk, now)
	check(t, err)
	v, err := trust.VerifySnapshot(a, snapshot, now)
	check(t, err)
	b := trust.EnvelopeBinding{Cluster: a.Cluster(), Node: nid, Lease: id(t), Job: id(t), Run: id(t), Worker: id(t), Attempt: id(t), Audience: "integration", CredentialKind: "synthetic_v1", IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix()}
	nodeChain := [][]byte{node, issuer, root}
	e, err := trust.SealCredentialEnvelope(a, v, [][]byte{controller, issuer, root}, ck, nodeChain, b, []byte("synthetic-secret"), now)
	check(t, err)
	digest, err := trust.EnvelopeDigest(e)
	check(t, err)
	j, err := nodecredentials.InitializeReplay(p, b, digest)
	check(t, err)
	opener, err := store.EnvelopeOpener(h)
	check(t, err)
	wrong := b
	wrong.Lease = id(t)
	if _, err := trust.OpenCredentialEnvelope(a, v, e, wrong, digest, cid, nodeChain, opener, j, now); err == nil {
		t.Fatal("wrong job accepted")
	}
	if yes, err := j.Reserved(); err != nil || yes {
		t.Fatal("unauthenticated reservation", err)
	}
	secret, err := trust.OpenCredentialEnvelope(a, v, e, b, digest, cid, nodeChain, opener, j, now)
	check(t, err)
	defer clear(secret)
	if string(secret) != "synthetic-secret" {
		t.Fatal("wrong plaintext")
	}
	j, err = nodecredentials.OpenReplay(p, b, digest)
	check(t, err)
	if _, err := trust.OpenCredentialEnvelope(a, v, e, b, digest, cid, nodeChain, opener, j, now); err == nil {
		t.Fatal("replayed envelope opened after restart")
	}
}
