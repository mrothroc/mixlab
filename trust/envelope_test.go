package trust

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

type envelopeFixture struct {
	trustFixture
	controller testPrincipal
	node       testPrincipal
	key        *securekeys.EnvelopeOpener
	binding    EnvelopeBinding
	envelope   CredentialEnvelope
	digest     string
}

func newEnvelopeFixture(t *testing.T) envelopeFixture {
	t.Helper()
	f := envelopeFixture{trustFixture: fixture(t)}
	f.view = f.snapshot(t, 1, testNow, nil)
	f.controller = f.principal(t, Controller)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	requireOK(t, err)
	requireOK(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	requireOK(t, err)
	store, err := securekeys.OpenFile(p, f.anchor.Fingerprint())
	requireOK(t, err)
	handle, err := store.GenerateEnvelope()
	requireOK(t, err)
	t.Cleanup(func() { requireOK(t, store.Delete(handle)); requireOK(t, store.Close()) })
	f.key, err = store.EnvelopeOpener(handle)
	requireOK(t, err)
	k := newKey(t)
	nodeID := newID(t)
	der, err := certificates.IssueNode(f.anchor, f.issuer, f.issuerKey, nodeID, k.Public(), handle.PublicKey, testNow)
	requireOK(t, err)
	_, identity, err := certificates.Parse(der)
	requireOK(t, err)
	f.node = testPrincipal{chain: [][]byte{der, f.issuer, f.root}, key: k, id: identity}
	f.binding = EnvelopeBinding{Cluster: f.anchor.Cluster(), Node: nodeID, Lease: newID(t), Job: newID(t), Run: newID(t), Worker: newID(t), Attempt: newID(t), Audience: "test/application", CredentialKind: "opaque_test_v1", IssuedAt: testNow.Unix(), ExpiresAt: testNow.Add(5 * time.Minute).Unix()}
	f.envelope, err = SealCredentialEnvelope(f.anchor, f.view, f.controller.chain, f.controller.key, f.node.chain, f.binding, []byte("SYNTHETIC-APPLICATION-SECRET"), testNow)
	requireOK(t, err)
	f.digest, err = EnvelopeDigest(f.envelope)
	requireOK(t, err)
	return f
}

type testReplay struct {
	used  map[string]bool
	calls int
	fail  bool
}

func (r *testReplay) ReserveEnvelope(_ EnvelopeBinding, digest string) error {
	r.calls++
	if r.fail || r.used[digest] {
		return errors.New("reservation rejected")
	}
	if r.used == nil {
		r.used = map[string]bool{}
	}
	r.used[digest] = true
	return nil
}

type countingOpener struct {
	EnvelopeKeyOpener
	calls int
}

func (o *countingOpener) Open(info, aad, enc, ct []byte) ([]byte, error) {
	o.calls++
	return o.EnvelopeKeyOpener.Open(info, aad, enc, ct)
}

func TestCredentialEnvelopeRoundTripAndReplay(t *testing.T) {
	f := newEnvelopeFixture(t)
	raw, err := canonical(f.envelope)
	requireOK(t, err)
	if bytes.Contains(raw, []byte("SYNTHETIC-APPLICATION-SECRET")) {
		t.Fatal("plaintext in transport")
	}
	e, err := DecodeEnvelope(raw)
	requireOK(t, err)
	key := &countingOpener{EnvelopeKeyOpener: f.key}
	replay := &testReplay{}
	got, err := OpenCredentialEnvelope(f.anchor, f.view, e, f.binding, f.digest, f.controller.id.Principal, f.node.chain, key, replay, testNow)
	requireOK(t, err)
	defer clear(got)
	if string(got) != "SYNTHETIC-APPLICATION-SECRET" || key.calls != 1 || replay.calls != 1 {
		t.Fatal("round trip")
	}
	if _, err := OpenCredentialEnvelope(f.anchor, f.view, e, f.binding, f.digest, f.controller.id.Principal, f.node.chain, key, replay, testNow); err == nil || key.calls != 1 {
		t.Fatal("replay decrypted")
	}
	second, err := SealCredentialEnvelope(f.anchor, f.view, f.controller.chain, f.controller.key, f.node.chain, f.binding, []byte("second"), testNow)
	requireOK(t, err)
	if second.Payload.Nonce == e.Payload.Nonce || bytes.Equal(second.Payload.Encapsulation, e.Payload.Encapsulation) {
		t.Fatal("nonce or ephemeral key reused")
	}
}

func TestCredentialEnvelopeRejectsBeforeDecryption(t *testing.T) {
	f := newEnvelopeFixture(t)
	for _, name := range []string{"cluster", "node", "lease", "job", "run", "worker", "attempt", "audience", "kind", "issue", "expiry", "manifest", "sender", "ciphertext", "recipient-key", "suite", "version", "nonce", "signature", "revoked-node", "revoked-controller", "expired", "stale", "nil-guard", "guard-error", "wrong-key"} {
		t.Run(name, func(t *testing.T) {
			raw, err := canonical(f.envelope)
			requireOK(t, err)
			e, err := DecodeEnvelope(raw)
			requireOK(t, err)
			want, digest, sender, now, view := f.binding, f.digest, f.controller.id.Principal, testNow, f.view
			r := &testReplay{}
			var replay EnvelopeReplayGuard = r
			key := &countingOpener{EnvelopeKeyOpener: f.key}
			switch name {
			case "cluster":
				want.Cluster = newID(t)
			case "node":
				want.Node = newID(t)
			case "lease":
				want.Lease = newID(t)
			case "job":
				want.Job = newID(t)
			case "run":
				want.Run = newID(t)
			case "worker":
				want.Worker = newID(t)
			case "attempt":
				want.Attempt = newID(t)
			case "audience":
				want.Audience += "/wrong"
			case "kind":
				want.CredentialKind += "/wrong"
			case "issue":
				want.IssuedAt++
			case "expiry":
				want.ExpiresAt++
			case "manifest":
				digest = strings.Repeat("0", 64)
			case "sender":
				sender = newID(t)
			case "ciphertext":
				e.Payload.Ciphertext[0] ^= 1
			case "recipient-key":
				e.Payload.RecipientKey[0] ^= 1
			case "suite":
				e.Payload.Suite = "other"
			case "version":
				e.Payload.Version = "v0"
			case "nonce":
				e.Payload.Nonce = strings.Repeat("AB", 32)
			case "signature":
				e.Proof.Signature[0] ^= 1
			case "revoked-node", "revoked-controller":
				principal := f.node.id.Principal
				if name == "revoked-controller" {
					principal = f.controller.id.Principal
				}
				view = f.snapshot(t, 2, testNow, []Revocation{{Kind: "principal", ID: principal, Mode: "prospective", Reason: "test", FirstGeneration: 2}})
			case "expired":
				now = time.Unix(want.ExpiresAt, 0)
			case "stale":
				now = testNow.Add(SnapshotLifetime)
			case "nil-guard":
				replay = nil
			case "guard-error":
				r.fail = true
			case "wrong-key":
				other := newEnvelopeFixture(t)
				key.EnvelopeKeyOpener = other.key
			}
			// Recompute manifest digest for wire mutations to exercise signature,
			// recipient and encoding checks, not only the manifest hash check.
			if name == "ciphertext" || name == "recipient-key" || name == "suite" || name == "version" || name == "nonce" || name == "signature" {
				digest, err = EnvelopeDigest(e)
				requireOK(t, err)
			}
			got, err := OpenCredentialEnvelope(f.anchor, view, e, want, digest, sender, f.node.chain, key, replay, now)
			if err == nil || len(got) != 0 || key.calls != 0 {
				t.Fatal("rejected envelope reached decryption", err)
			}
			if name != "guard-error" && r.calls != 0 {
				t.Fatal("invalid envelope reached journal")
			}
		})
	}
}

func TestCredentialEnvelopeClosedPurposeAndDecode(t *testing.T) {
	f := newEnvelopeFixture(t)
	for _, role := range []Role{Authority, Coordinator} {
		p := f.principal(t, role)
		if _, err := SealCredentialEnvelope(f.anchor, f.view, p.chain, p.key, f.node.chain, f.binding, []byte("opaque"), testNow); err == nil {
			t.Fatal("wrong sealing role", role)
		}
	}
	for _, secret := range [][]byte{nil, make([]byte, 65537)} {
		if _, err := SealCredentialEnvelope(f.anchor, f.view, f.controller.chain, f.controller.key, f.node.chain, f.binding, secret, testNow); err == nil {
			t.Fatal("bad payload size")
		}
	}
	raw, err := json.Marshal(f.envelope)
	requireOK(t, err)
	for _, b := range [][]byte{nil, []byte("{"), append([]byte(" "), raw...), append(raw[:len(raw)-1], []byte(",\"unknown\":0}")...)} {
		if _, err := DecodeEnvelope(b); err == nil {
			t.Fatal("noncanonical encoding accepted")
		}
	}
}

func TestWorkloadExactAdmissionAndRevocation(t *testing.T) {
	f := newEnvelopeFixture(t)
	b := WorkloadBinding{Version: WorkloadBindingVersion, Cluster: f.anchor.Cluster(), Role: Worker, Principal: newID(t), Participant: newID(t), Run: newID(t), Job: newID(t), Attempt: newID(t), Audience: "ddp/data", IssuedAt: testNow.Unix(), ExpiresAt: testNow.Add(time.Hour).Unix(), Group: newID(t), Generation: 2, MembershipHash: strings.Repeat("ab", 32), Member: newID(t), Rank: 1}
	b.Lease, b.ManifestHash = newID(t), strings.Repeat("ef", 32)
	der, err := certificates.IssueWorkload(f.anchor, f.issuer, f.issuerKey, newKey(t).Public(), b, testNow.Add(time.Hour), testNow)
	requireOK(t, err)
	chain := [][]byte{der, f.issuer, f.root}
	requireOK(t, VerifyWorkload(f.anchor, f.view, chain, b, testNow))
	for _, name := range []string{"rank", "attempt", "membership", "manifest", "lease", "group", "generation", "audience"} {
		want := b
		switch name {
		case "rank":
			want.Rank++
		case "attempt":
			want.Attempt = newID(t)
		case "membership":
			want.MembershipHash = strings.Repeat("cd", 32)
		case "manifest":
			want.ManifestHash = strings.Repeat("cd", 32)
		case "lease":
			want.Lease = newID(t)
		case "group":
			want.Group = newID(t)
		case "generation":
			want.Generation++
		case "audience":
			want.Audience = "other"
		}
		if err := VerifyWorkload(f.anchor, f.view, chain, want, testNow); err == nil {
			t.Fatal("wrong admitted", name)
		}
	}
	v := f.snapshot(t, 2, testNow, []Revocation{{Kind: "principal", ID: b.Principal, Mode: "compromise", Reason: "test", FirstGeneration: 2}})
	if err := VerifyWorkload(f.anchor, v, chain, b, testNow); err == nil {
		t.Fatal("revoked worker accepted")
	}
	if _, err := NodeEnvelopeKey(f.anchor, f.view, chain, b.Principal, testNow); err == nil {
		t.Fatal("worker used as node")
	}
}
