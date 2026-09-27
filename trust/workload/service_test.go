package workload

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ecdh"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type countingSigner struct {
	crypto.Signer
	calls int
}

func (s *countingSigner) Sign(r io.Reader, b []byte, opts crypto.SignerOpts) ([]byte, error) {
	s.calls++
	return s.Signer.Sign(r, b, opts)
}

type fixture struct {
	now                    time.Time
	path                   statehome.Path
	a                      Authority
	issuer                 *countingSigner
	view                   trust.VerifiedSnapshot
	snapshot               trust.SignedSnapshot
	snapshotCert           []byte
	snapshotKey            crypto.Signer
	node, controller       [][]byte
	nodeKey, controllerKey crypto.Signer
	grant                  Grant
	s                      *Service
}

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func randomID(t *testing.T) string {
	t.Helper()
	b := make([]byte, 16)
	_, err := rand.Read(b)
	check(t, err)
	return hex.EncodeToString(b)
}
func newKey(t *testing.T) crypto.Signer {
	t.Helper()
	_, key, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	return key
}
func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{now: time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)}
	rootKey := newKey(t)
	root, err := certificates.CreateRoot(randomID(t), rootKey, f.now)
	check(t, err)
	fingerprint, err := trust.RootFingerprint(rootKey.Public())
	check(t, err)
	f.a.Anchor, err = trust.PinRoot(root, fingerprint, f.now)
	check(t, err)
	f.issuer = &countingSigner{Signer: newKey(t)}
	f.a.Key = f.issuer
	f.a.Issuer, err = certificates.Issue(f.a.Anchor, root, rootKey, certificates.Issuer, "", "", f.issuer.Public(), f.now)
	check(t, err)
	f.snapshotKey = newKey(t)
	f.snapshotCert, err = certificates.Issue(f.a.Anchor, root, rootKey, certificates.SnapshotSigner, "", "", f.snapshotKey.Public(), f.now)
	check(t, err)
	endpoints, err := trust.SignAuthorityEndpoints(f.a.Anchor, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: f.a.Anchor.Cluster(), Audience: "test", URLs: []string{"https://test.invalid"}, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(24 * time.Hour).Unix()}, rootKey, f.now)
	check(t, err)
	f.snapshot.Payload = trust.Snapshot{Version: trust.SnapshotVersion, Cluster: f.a.Anchor.Cluster(), Generation: 1, IssuedAt: f.now.Unix(), ExpiresAt: f.now.Add(trust.SnapshotLifetime).Unix(), Issuers: [][]byte{f.a.Issuer}, EligibleRoles: []trust.Role{trust.Controller, trust.Node, trust.Worker}, Endpoints: endpoints}
	f.refresh(t)
	node, controller := randomID(t), randomID(t)
	f.nodeKey, f.controllerKey = newKey(t), newKey(t)
	envelope, err := ecdh.X25519().GenerateKey(rand.Reader)
	check(t, err)
	der, err := certificates.IssueNode(f.a.Anchor, f.a.Issuer, f.issuer, node, f.nodeKey.Public(), envelope.PublicKey().Bytes(), f.now)
	check(t, err)
	f.node = [][]byte{der, f.a.Issuer, root}
	der, err = certificates.Issue(f.a.Anchor, f.a.Issuer, f.issuer, certificates.Principal, trust.Controller, controller, f.controllerKey.Public(), f.now)
	check(t, err)
	f.controller = [][]byte{der, f.a.Issuer, root}
	scope := Scope{Cluster: f.a.Anchor.Cluster(), Controller: controller, Node: node, Lease: randomID(t), Run: randomID(t), Job: randomID(t), Attempt: randomID(t), Group: randomID(t), Member: randomID(t), Workload: randomID(t), Generation: 1, Rank: 0, MembershipHash: strings.Repeat("a", 64), ManifestHash: strings.Repeat("b", 64), Created: f.now.Unix(), AdmitUntil: f.now.Add(time.Hour).Unix(), Deadline: f.now.Add(3 * 24 * time.Hour).Unix()}
	f.grant = f.sign(t, scope, newKey(t))
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	f.path, err = statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Authority})
	check(t, err)
	f.s, err = Initialize(context.Background(), f.path, f.a, f.view, f.now)
	check(t, err)
	f.issuer.calls = 0
	return f
}
func (f *fixture) refresh(t *testing.T) {
	t.Helper()
	var err error
	f.snapshot, err = trust.SignSnapshot(f.a.Anchor, f.snapshot.Payload, f.snapshotCert, f.snapshotKey, f.now)
	check(t, err)
	f.view, err = trust.VerifySnapshot(f.a.Anchor, f.snapshot, f.now)
	check(t, err)
}
func (f *fixture) sign(t *testing.T, s Scope, key crypto.Signer) Grant {
	t.Helper()
	r, err := NewRequest(s, key)
	check(t, err)
	q, err := r.SigningRequest()
	check(t, err)
	p, err := trust.SignPrincipalProof(f.a.Anchor, f.node, f.nodeKey, f.view, q, f.now)
	check(t, err)
	n := SignedRequest{r, p}
	q, err = n.GrantRequest()
	check(t, err)
	p, err = trust.SignPrincipalProof(f.a.Anchor, f.controller, f.controllerKey, f.view, q, f.now)
	check(t, err)
	return Grant{n, p}
}

func TestWorkloadIssueRetryAfterSnapshotExpiry(t *testing.T) {
	f := newFixture(t)
	out, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
	check(t, err)
	check(t, ValidateResult(f.a.Anchor, f.grant, out, f.now))
	if f.issuer.calls != 1 || out.Binding.ExpiresAt != f.grant.Request.Request.Scope.Deadline {
		t.Fatal("wrong issuance count/deadline", f.issuer.calls)
	}
	f.now = f.now.Add(20 * time.Minute)
	f.snapshot.Payload.Generation++
	f.snapshot.Payload.IssuedAt = f.now.Unix()
	f.snapshot.Payload.ExpiresAt = f.now.Add(trust.SnapshotLifetime).Unix()
	f.refresh(t)
	f.s, err = Open(f.path, f.a, f.view, f.now)
	check(t, err)
	again, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
	check(t, err)
	check(t, ValidateResult(f.a.Anchor, f.grant, again, f.now))
	if !bytes.Equal(out.Chain[0], again.Chain[0]) || f.issuer.calls != 1 || again.Snapshot.Payload.Generation != 2 {
		t.Fatal("retry reissued or failed to refresh public trust")
	}
}

func TestWorkloadRejectsTamperingAndPurposeReuse(t *testing.T) {
	for _, mode := range []string{"csr", "scope", "controller", "node-proof", "grant-proof", "expired", "node-key", "controller-key", "duplicate-key", "duplicate-job", "duplicate-principal"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			g, chain, now := f.grant, f.controller, f.now
			scope := g.Request.Request.Scope
			switch mode {
			case "csr":
				g.Request.Request.CSR[10] ^= 1
			case "scope":
				g.Request.Request.Scope.Rank++
			case "controller":
				chain = f.node
			case "node-proof":
				g.Request.Proof.Signature[0] ^= 1
			case "grant-proof":
				g.Proof.Signature[0] ^= 1
			case "expired":
				now = now.Add(time.Hour)
			case "node-key":
				g = f.sign(t, scope, f.nodeKey)
			case "controller-key":
				g = f.sign(t, scope, f.controllerKey)
			case "duplicate-key", "duplicate-job", "duplicate-principal":
				key := newKey(t)
				original := f.sign(t, scope, key)
				_, err := f.s.Issue(context.Background(), chain, original, f.view, now)
				check(t, err)
				if mode != "duplicate-job" {
					scope.Job = randomID(t)
				}
				if mode != "duplicate-principal" {
					scope.Workload = randomID(t)
				}
				if mode != "duplicate-key" {
					key = newKey(t)
				}
				g = f.sign(t, scope, key)
			}
			before := f.issuer.calls
			_, err := f.s.Issue(context.Background(), chain, g, f.view, now)
			if err == nil || f.issuer.calls != before {
				t.Fatal("invalid grant reached issuer", mode, err)
			}
		})
	}
}

func TestWorkloadRevocationBlocksCommittedRetry(t *testing.T) {
	for _, mode := range []string{"prospective", "compromise"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			_, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			check(t, err)
			f.snapshot.Payload.Generation++
			f.snapshot.Payload.Revocations = []trust.Revocation{{Kind: "principal", ID: f.grant.Request.Request.Scope.Node, Mode: mode, Reason: "test", FirstGeneration: 2}}
			f.refresh(t)
			_, err = f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			if err == nil || f.issuer.calls != 1 {
				t.Fatal("revoked node accepted", err)
			}
		})
	}
}

func TestWorkloadRetryRechecksCurrentCallerKeyPurpose(t *testing.T) {
	f := newFixture(t)
	key := newKey(t)
	f.grant = f.sign(t, f.grant.Request.Request.Scope, key)
	f.s.store = &failingPublication{journal: f.path}
	if _, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now); err == nil {
		t.Fatal("expected interruption before result publication")
	}
	f.s.store = f.path
	scope := f.grant.Request.Request.Scope
	der, err := certificates.Issue(f.a.Anchor, f.a.Issuer, f.issuer, certificates.Principal, trust.Controller, scope.Controller, key.Public(), f.now)
	check(t, err)
	caller := [][]byte{der, f.a.Issuer, f.a.Anchor.DER()}
	before := f.issuer.calls
	if _, err := f.s.Issue(context.Background(), caller, f.grant, f.view, f.now); err == nil || f.issuer.calls != before {
		t.Fatal("retry reused current caller key", err)
	}
}

func TestWorkloadRetryCannotReviveExpiredAdmission(t *testing.T) {
	for _, committed := range []bool{false, true} {
		f := newFixture(t)
		if !committed {
			f.s.store = &failingPublication{journal: f.path}
		}
		_, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
		if (err == nil) != committed {
			t.Fatal(err)
		}
		f.s.store = f.path
		f.now = time.Unix(f.grant.Request.Request.Scope.AdmitUntil, 0)
		f.snapshot.Payload.Generation++
		f.snapshot.Payload.IssuedAt = f.now.Unix()
		f.snapshot.Payload.ExpiresAt = f.now.Add(trust.SnapshotLifetime).Unix()
		f.refresh(t)
		before := f.issuer.calls
		if _, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now); err == nil || before != f.issuer.calls {
			t.Fatal("expired admission revived", err)
		}
	}
}

type failingPublication struct {
	journal
	after  bool
	failed bool
}

func (j *failingPublication) CompareAndSwap(name string, old, next []byte) error {
	if name == stateFile && !j.failed && bytes.Contains(next, []byte(`"result":{"grant_hash"`)) {
		j.failed = true
		if j.after {
			if err := j.journal.CompareAndSwap(name, old, next); err != nil {
				return err
			}
		}
		return errors.New("injected uncertain certificate publication")
	}
	return j.journal.CompareAndSwap(name, old, next)
}
func TestWorkloadIssuancePublicationRecovery(t *testing.T) {
	for _, after := range []bool{false, true} {
		t.Run(map[bool]string{false: "before", true: "after"}[after], func(t *testing.T) {
			f := newFixture(t)
			f.s.store = &failingPublication{journal: f.path, after: after}
			out, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			if err == nil || len(out.Chain) != 0 {
				t.Fatal("uncommitted certificate escaped")
			}
			f.s, err = Open(f.path, f.a, f.view, f.now)
			check(t, err)
			out, err = f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			check(t, err)
			check(t, ValidateResult(f.a.Anchor, f.grant, out, f.now))
			want := 2
			if after {
				want = 1
			}
			if f.issuer.calls != want {
				t.Fatal(f.issuer.calls)
			}
		})
	}
}

func TestWorkloadConcurrentRetryAndMissingHistory(t *testing.T) {
	f := newFixture(t)
	var wg sync.WaitGroup
	results := make(chan Result, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := f.s.Issue(context.Background(), f.controller, f.grant, f.view, f.now)
			if err != nil {
				t.Error(err)
				return
			}
			results <- out
		}()
	}
	wg.Wait()
	close(results)
	var first []byte
	for out := range results {
		if first == nil {
			first = out.Chain[0]
		}
		if !bytes.Equal(first, out.Chain[0]) {
			t.Fatal("concurrent retry changed certificate")
		}
	}
	if f.issuer.calls != 1 {
		t.Fatal(f.issuer.calls)
	}
	check(t, os.Remove(filepath.Join(f.path.Dir(), stateFile)))
	if _, err := Open(f.path, f.a, f.view, f.now); err == nil {
		t.Fatal("missing journal accepted")
	}
	if _, err := Initialize(context.Background(), f.path, f.a, f.view, f.now); err == nil {
		t.Fatal("missing published history recreated")
	}
}

func TestNodeWorkloadProofCannotGrantAdmission(t *testing.T) {
	f := newFixture(t)
	q, err := f.grant.Request.GrantRequest()
	check(t, err)
	if _, err := trust.SignPrincipalProof(f.a.Anchor, f.node, f.nodeKey, f.view, q, f.now); err == nil {
		t.Fatal("node granted admission")
	}
	q, err = f.grant.Request.Request.SigningRequest()
	check(t, err)
	if _, err := trust.SignPrincipalProof(f.a.Anchor, f.controller, f.controllerKey, f.view, q, f.now); err == nil {
		t.Fatal("controller impersonated node key request")
	}
}
