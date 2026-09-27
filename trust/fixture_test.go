package trust

import (
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

var testNow = time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)

func requireOK(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func newKey(t *testing.T) ed25519.PrivateKey {
	t.Helper()
	_, k, err := ed25519.GenerateKey(rand.Reader)
	requireOK(t, err)
	return k
}

func newID(t *testing.T) string {
	t.Helper()
	id, err := certificates.NewID()
	requireOK(t, err)
	return id
}

type testPrincipal struct {
	chain [][]byte
	key   ed25519.PrivateKey
	id    certificates.Identity
}

type trustFixture struct {
	anchor                        Anchor
	root, issuer, signer          []byte
	rootKey, issuerKey, signerKey ed25519.PrivateKey
	endpoints                     SignedEndpoints
	view                          VerifiedSnapshot
}

func fixture(t *testing.T) trustFixture {
	t.Helper()
	f := trustFixture{rootKey: newKey(t), issuerKey: newKey(t), signerKey: newKey(t)}
	var err error
	f.root, err = certificates.CreateRoot(newID(t), f.rootKey, testNow)
	requireOK(t, err)
	fp, err := RootFingerprint(f.rootKey.Public())
	requireOK(t, err)
	f.anchor, err = PinRoot(f.root, fp, testNow)
	requireOK(t, err)
	f.issuer, err = certificates.Issue(f.anchor, f.root, f.rootKey, certificates.Issuer, "", "", f.issuerKey.Public(), testNow)
	requireOK(t, err)
	f.signer, err = certificates.Issue(f.anchor, f.root, f.rootKey, certificates.SnapshotSigner, "", "", f.signerKey.Public(), testNow)
	requireOK(t, err)
	f.endpoints, err = SignAuthorityEndpoints(f.anchor, AuthorityEndpoints{
		Version: EndpointVersion, Cluster: f.anchor.Cluster(), Audience: "authority-service",
		URLs: []string{"https://authority.example:7443"}, IssuedAt: testNow.Unix(), ExpiresAt: testNow.Add(180 * 24 * time.Hour).Unix(),
	}, f.rootKey, testNow)
	requireOK(t, err)
	f.view = f.snapshot(t, 1, testNow, nil)
	return f
}

func (f trustFixture) snapshot(t *testing.T, gen uint64, now time.Time, revocations []Revocation) VerifiedSnapshot {
	t.Helper()
	s, err := SignSnapshot(f.anchor, Snapshot{
		Version: SnapshotVersion, Cluster: f.anchor.Cluster(), Generation: gen,
		IssuedAt: now.Unix(), ExpiresAt: now.Add(SnapshotLifetime).Unix(),
		Issuers: [][]byte{f.issuer}, EligibleRoles: []Role{Authority, Controller, Coordinator, Node, Worker},
		Endpoints: f.endpoints, Revocations: revocations,
	}, f.signer, f.signerKey, now)
	requireOK(t, err)
	v, err := VerifySnapshot(f.anchor, s, now)
	requireOK(t, err)
	return v
}

func (f trustFixture) principal(t *testing.T, role Role) testPrincipal {
	t.Helper()
	k := newKey(t)
	der, err := certificates.Issue(f.anchor, f.issuer, f.issuerKey, certificates.Principal, role, newID(t), k.Public(), testNow)
	requireOK(t, err)
	_, id, err := certificates.Parse(der)
	requireOK(t, err)
	return testPrincipal{chain: [][]byte{der, f.issuer, f.root}, key: k, id: id}
}

func request(purpose Purpose) SignRequest {
	return SignRequest{Version: ProofVersion, Purpose: purpose, Digest: strings.Repeat("ab", 32), Context: "job/attempt-1", Audience: "node/receiver-1"}
}

type countingSigner struct {
	crypto.Signer
	calls int
}

func (s *countingSigner) Sign(r io.Reader, digest []byte, opts crypto.SignerOpts) ([]byte, error) {
	s.calls++
	return s.Signer.Sign(r, digest, opts)
}

type memoryJournal map[string]uint64

func (j memoryJournal) AcceptedGeneration(d string) (uint64, bool, error) {
	g, ok := j[d]
	return g, ok, nil
}

func cloneSnapshot(t *testing.T, s SignedSnapshot) SignedSnapshot {
	t.Helper()
	b, err := canonical(s)
	requireOK(t, err)
	var out SignedSnapshot
	requireOK(t, decodeCanonical(b, &out))
	return out
}

func cloneProof(t *testing.T, p SignedProof) SignedProof {
	t.Helper()
	b, err := canonical(p)
	requireOK(t, err)
	q, err := DecodeProof(b)
	requireOK(t, err)
	return q
}
