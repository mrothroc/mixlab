package enrollment

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

var now = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
var ctx = context.Background()

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
func id(t *testing.T) string { t.Helper(); s, err := certificates.NewID(); check(t, err); return s }
func path(t *testing.T, kind statehome.Kind) statehome.Path {
	t.Helper()
	d, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(d, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: d}, statehome.Context{Kind: kind})
	check(t, err)
	return p
}

type countingSigner struct {
	crypto.Signer
	calls int
}

func (s *countingSigner) Sign(r io.Reader, b []byte, o crypto.SignerOpts) ([]byte, error) {
	s.calls++
	return s.Signer.Sign(r, b, o)
}

type fixture struct {
	p            statehome.Path
	a            Authority
	s            *Service
	v            trust.VerifiedSnapshot
	snapshotCert []byte
	snapshotKey  ed25519.PrivateKey
	endpoints    trust.SignedEndpoints
	issuer       *countingSigner
}

func setup(t *testing.T) fixture {
	t.Helper()
	f := fixture{p: path(t, statehome.Authority), snapshotKey: key(t)}
	rk, ik, ak := key(t), key(t), key(t)
	root, err := certificates.CreateRoot(id(t), rk, now)
	check(t, err)
	fp, err := trust.RootFingerprint(rk.Public())
	check(t, err)
	f.a.Anchor, err = trust.PinRoot(root, fp, now)
	check(t, err)
	f.a.Issuer, err = certificates.Issue(f.a.Anchor, root, rk, certificates.Issuer, "", "", ik.Public(), now)
	check(t, err)
	f.snapshotCert, err = certificates.Issue(f.a.Anchor, root, rk, certificates.SnapshotSigner, "", "", f.snapshotKey.Public(), now)
	check(t, err)
	leaf, err := certificates.Issue(f.a.Anchor, f.a.Issuer, ik, certificates.Principal, certificates.Authority, id(t), ak.Public(), now)
	check(t, err)
	f.issuer = &countingSigner{Signer: ik}
	f.a.IssuerKey, f.a.PrincipalKey, f.a.PrincipalChain = f.issuer, ak, [][]byte{leaf, f.a.Issuer, root}
	f.endpoints, err = trust.SignAuthorityEndpoints(f.a.Anchor, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: f.a.Anchor.Cluster(),
		Audience: "authority", URLs: []string{"https://authority.example:7443"}, IssuedAt: now.Unix(), ExpiresAt: now.Add(24 * time.Hour).Unix()}, rk, now)
	check(t, err)
	f.v = f.view(t, 1, now, nil)
	f.s, err = Initialize(ctx, f.p, f.a, f.v, now)
	check(t, err)
	return f
}
func (f fixture) view(t *testing.T, gen uint64, at time.Time, rev []trust.Revocation) trust.VerifiedSnapshot {
	t.Helper()
	s, err := trust.SignSnapshot(f.a.Anchor, trust.Snapshot{Version: trust.SnapshotVersion, Cluster: f.a.Anchor.Cluster(), Generation: gen, IssuedAt: at.Unix(), ExpiresAt: at.Add(trust.SnapshotLifetime).Unix(),
		Issuers: [][]byte{f.a.Issuer}, EligibleRoles: []trust.Role{trust.Authority, trust.Controller, trust.Coordinator, trust.Node, trust.Worker}, Endpoints: f.endpoints, Revocations: rev}, f.snapshotCert, f.snapshotKey, at)
	check(t, err)
	v, err := trust.VerifySnapshot(f.a.Anchor, s, at)
	check(t, err)
	return v
}
func (f fixture) invite(t *testing.T, p Purpose) Invitation {
	t.Helper()
	i, err := f.s.Invite(ctx, p, "https://authority.example:7443", "authority", 10*time.Minute, f.v, now)
	check(t, err)
	return i
}
func requestFor(t *testing.T, i Invitation) SignedRequest {
	t.Helper()
	p := path(t, statehome.Principal)
	store, err := securekeys.OpenFile(p, strings.Repeat("ab", 32))
	check(t, err)
	t.Cleanup(func() { check(t, store.Close()) })
	h, err := store.Generate()
	check(t, err)
	signer, err := store.Signer(h)
	check(t, err)
	var public []byte
	if i.Role == trust.Node {
		e, err := store.GenerateEnvelope()
		check(t, err)
		public = e.PublicKey
	}
	r, err := NewRequest(i, signer, public, now)
	check(t, err)
	return r
}

func TestProvisionedEnrollmentAllPrincipalRoles(t *testing.T) {
	for _, purpose := range []Purpose{NodeEnrollment, ControllerEnrollment, CoordinatorEnrollment} {
		t.Run(string(purpose), func(t *testing.T) {
			f := setup(t)
			i := f.invite(t, purpose)
			defer i.Clear()
			q := requestFor(t, i)
			out, err := f.s.Consume(ctx, q, i.Secret, f.v, now)
			check(t, err)
			_, err = ValidateResult(i, q, out, now)
			check(t, err)
			if f.issuer.calls != 1 {
				t.Fatal("unexpected issuance count", f.issuer.calls)
			}
			if _, err := f.s.Consume(ctx, q, i.Secret, f.v, now); !errors.Is(err, ErrConsumed) {
				t.Fatal("replay accepted", err)
			}
			reopened, err := Open(f.p, f.a, f.v, now)
			check(t, err)
			again, err := reopened.RecoverApproved(ctx, i.ID, f.v, now)
			check(t, err)
			if !bytes.Equal(again.Chain[0], out.Chain[0]) || f.issuer.calls != 1 {
				t.Fatal("recovery reissued identity")
			}
			raw, err := f.p.ReadFile(stateFile)
			check(t, err)
			encodedSecret, _ := json.Marshal(i.Secret)
			if bytes.Contains(raw, encodedSecret) {
				t.Fatal("authority persisted bearer secret")
			}
			if strings.Contains(fmt.Sprintf("%+v %#v", i, i), string(encodedSecret)) {
				t.Fatal("invitation formatting disclosed secret")
			}
		})
	}
}

func TestProvisionedConcurrentConsumeOneWinner(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	q := requestFor(t, i)
	var won atomic.Int32
	var wg sync.WaitGroup
	for n := 0; n < 8; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := f.s.Consume(ctx, q, i.Secret, f.v, now)
			if err == nil {
				won.Add(1)
			} else if !errors.Is(err, ErrConsumed) {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if won.Load() != 1 || f.issuer.calls != 1 {
		t.Fatalf("winners=%d issued=%d", won.Load(), f.issuer.calls)
	}
}

type failingJournal struct {
	journal
	stage string
	after bool
	fired bool
}

func (j *failingJournal) CompareAndSwap(name string, old, next []byte) error {
	var r record
	if err := json.Unmarshal(next, &r); err != nil {
		return err
	}
	match := false
	for _, e := range r.Entries {
		if e.Stage == j.stage {
			match = true
		}
	}
	for _, e := range r.Interactive {
		if e.Stage == j.stage {
			match = true
		}
	}
	for _, e := range r.Renewals {
		if e.Stage == j.stage {
			match = true
		}
	}
	if !j.fired && match {
		j.fired = true
		if j.after {
			if err := j.journal.CompareAndSwap(name, old, next); err != nil {
				return err
			}
		}
		return fmt.Errorf("injected publication failure")
	}
	return j.journal.CompareAndSwap(name, old, next)
}

func TestProvisionedPublicationRecovery(t *testing.T) {
	for _, stage := range []string{"approved", "issued"} {
		for _, after := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/after=%v", stage, after), func(t *testing.T) {
				f := setup(t)
				i := f.invite(t, NodeEnrollment)
				defer i.Clear()
				q := requestFor(t, i)
				f.s.store = &failingJournal{journal: f.p, stage: stage, after: after}
				out, err := f.s.Consume(ctx, q, i.Secret, f.v, now)
				if err == nil || len(out.Chain) != 0 {
					t.Fatal("uncertain publication returned credential")
				}
				reopen, err := Open(f.p, f.a, f.v, now)
				check(t, err)
				if stage == "approved" && !after {
					if f.issuer.calls != 0 {
						t.Fatal("issuer used before durable approval")
					}
					if _, err := reopen.RecoverApproved(ctx, i.ID, f.v, now); err == nil {
						t.Fatal("recovered nonexistent approval")
					}
					_, err := reopen.Consume(ctx, q, i.Secret, f.v, now)
					check(t, err)
					return
				}
				_, r, err := reopen.read()
				check(t, err)
				principal := r.Entries[0].Result.Approval.Principal
				out, err = reopen.RecoverApproved(ctx, i.ID, f.v, now)
				check(t, err)
				if out.Approval.Principal != principal {
					t.Fatal("recovery replaced approved identity")
				}
				_, err = ValidateResult(i, q, out, now)
				check(t, err)
				if stage == "issued" && after && f.issuer.calls != 1 {
					t.Fatal("durable issued leaf regenerated")
				}
				if _, err := reopen.Consume(ctx, q, i.Secret, f.v, now); !errors.Is(err, ErrConsumed) {
					t.Fatal("recovered invitation reused")
				}
			})
		}
	}
}

func TestProvisionedRejectsBeforeIssuance(t *testing.T) {
	for _, name := range []string{"secret", "short-secret", "purpose", "role", "audience", "endpoint", "root", "signature", "expiry"} {
		t.Run(name, func(t *testing.T) {
			f := setup(t)
			i := f.invite(t, NodeEnrollment)
			defer i.Clear()
			q := requestFor(t, i)
			secret := bytes.Clone(i.Secret)
			at := now
			switch name {
			case "secret":
				secret[0] ^= 1
			case "short-secret":
				secret = secret[:2]
			case "purpose":
				q.Request.Purpose = ControllerEnrollment
			case "role":
				q.Request.Role = trust.Controller
			case "audience":
				q.Request.Audience = "other"
			case "endpoint":
				q.Request.Endpoint = "https://other.example"
			case "root":
				q.Request.Fingerprint = strings.Repeat("00", 32)
			case "signature":
				q.Signature[0] ^= 1
			case "expiry":
				at = now.Add(10 * time.Minute)
			}
			if _, err := f.s.Consume(ctx, q, secret, f.v, at); err == nil {
				t.Fatal("invalid consume accepted")
			}
			if f.issuer.calls != 0 {
				t.Fatal("invalid consume reached issuer")
			}
			_, r, err := f.s.read()
			check(t, err)
			if r.Entries[0].Stage != "available" {
				t.Fatal("invalid proof spent invitation")
			}
		})
	}
}

func TestProvisionedStateFailsClosed(t *testing.T) {
	f := setup(t)
	if _, err := Initialize(ctx, f.p, f.a, f.v, now); err == nil {
		t.Fatal("reset enrollment journal")
	}
	newPath := path(t, statehome.Authority)
	if _, err := Open(newPath, f.a, f.v, now); err == nil {
		t.Fatal("missing state opened")
	}
	check(t, f.p.WriteFile(stateFile, []byte(`{}`)))
	if _, err := Open(f.p, f.a, f.v, now); err == nil {
		t.Fatal("corrupt state opened")
	}
	if _, err := f.s.Invite(ctx, NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, f.v, now); err == nil {
		t.Fatal("corrupt state reinitialized")
	}
}
