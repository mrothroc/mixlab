package authority

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

var ctx = context.Background()
var now = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)

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
func newID(t *testing.T) string { t.Helper(); v, err := certificates.NewID(); check(t, err); return v }

func fixture(t *testing.T) (*Store, statehome.Path) {
	t.Helper()
	r, i, sk := key(t), key(t), key(t)
	root, err := certificates.CreateRoot(newID(t), r, now)
	check(t, err)
	fp, err := trust.RootFingerprint(r.Public())
	check(t, err)
	a, err := trust.PinRoot(root, fp, now)
	check(t, err)
	issuer, err := certificates.Issue(a, root, r, certificates.Issuer, "", "", i.Public(), now)
	check(t, err)
	signer, err := certificates.Issue(a, root, r, certificates.SnapshotSigner, "", "", sk.Public(), now)
	check(t, err)
	eps, err := trust.SignAuthorityEndpoints(a, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: a.Cluster(), Audience: "test", URLs: []string{"https://test.invalid"}, IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Hour).Unix()}, r, now)
	check(t, err)
	snapshot, err := trust.SignSnapshot(a, trust.Snapshot{Version: trust.SnapshotVersion, Cluster: a.Cluster(), Generation: 1, IssuedAt: now.Unix(), ExpiresAt: now.Add(trust.SnapshotLifetime).Unix(), Issuers: [][]byte{issuer}, EligibleRoles: []trust.Role{trust.Authority, trust.Controller, trust.Node}, Endpoints: eps}, signer, sk, now)
	check(t, err)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Authority})
	check(t, err)
	s, err := Initialize(ctx, p, a, signer, sk, snapshot, now)
	check(t, err)
	return s, p
}

func TestRefreshAndDurableRevocations(t *testing.T) {
	s, p := fixture(t)
	initial, err := s.Load(now)
	check(t, err)
	fresh, err := s.Refresh(ctx, now.Add(time.Minute))
	check(t, err)
	if !reflect.DeepEqual(initial, fresh) {
		t.Fatal("unnecessary refresh")
	}
	target := newID(t)
	revoked, err := s.Revoke(ctx, "principal", target, "prospective", "operator request", now)
	check(t, err)
	if revoked.Payload.Generation != 2 || revoked.Payload.Revocations[0].FirstGeneration != 2 {
		t.Fatal(revoked.Payload)
	}
	retry, err := s.Revoke(ctx, "principal", target, "prospective", "operator request", now)
	check(t, err)
	if !reflect.DeepEqual(revoked, retry) {
		t.Fatal("retry changed history")
	}
	s, err = Open(p, s.anchor, s.signerCertificate, s.signer, now.Add(time.Hour/2))
	check(t, err)
	fresh, err = s.Refresh(ctx, now.Add(time.Hour/2))
	check(t, err)
	if fresh.Payload.Generation != 3 || !reflect.DeepEqual(fresh.Payload.Revocations, revoked.Payload.Revocations) {
		t.Fatal("refresh lost revocation")
	}
	_, err = trust.VerifySnapshot(s.anchor, fresh, now.Add(time.Hour/2))
	check(t, err)
	escalated, err := s.Revoke(ctx, "principal", target, "compromise", "key compromised", now.Add(time.Hour/2))
	check(t, err)
	if escalated.Payload.Generation != 4 || escalated.Payload.Revocations[0].FirstGeneration != 2 {
		t.Fatal("escalation rewrote first generation")
	}
	if _, err := s.Revoke(ctx, "principal", target, "prospective", "undo", now.Add(time.Hour/2)); err == nil {
		t.Fatal("downgrade accepted")
	}
	if _, err := s.Refresh(ctx, now.Add(time.Hour)); err == nil {
		t.Fatal("extended expired root-signed endpoints")
	}
}

func TestSnapshotAuthorityConcurrentRefresh(t *testing.T) {
	s, p := fixture(t)
	other, err := Open(p, s.anchor, s.signerCertificate, s.signer, now)
	check(t, err)
	var wg sync.WaitGroup
	results := make(chan trust.SignedSnapshot, 2)
	errs := make(chan error, 2)
	for _, store := range []*Store{s, other} {
		wg.Add(1)
		go func(s *Store) {
			defer wg.Done()
			v, err := s.Refresh(ctx, now.Add(11*time.Minute))
			results <- v
			errs <- err
		}(store)
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		check(t, err)
	}
	for result := range results {
		if result.Payload.Generation != 2 {
			t.Fatal("competing refreshes", result.Payload.Generation)
		}
	}
}

type faultStore struct {
	statehome.Path
	after bool
}

var errFault = errors.New("injected publication failure")

func (f faultStore) CompareAndSwap(name string, old, next []byte) error {
	if f.after {
		if err := f.Path.CompareAndSwap(name, old, next); err != nil {
			return err
		}
	}
	return errFault
}

func TestRevocationPublicationFailure(t *testing.T) {
	for _, after := range []bool{false, true} {
		t.Run(map[bool]string{false: "before", true: "after"}[after], func(t *testing.T) {
			s, p := fixture(t)
			s.store = faultStore{p, after}
			target := newID(t)
			result, err := s.Revoke(ctx, "principal", target, "compromise", "test", now)
			if !errors.Is(err, errFault) || result.Payload.Generation != 0 {
				t.Fatal("uncertain success escaped", result, err)
			}
			reopened, err := Open(p, s.anchor, s.signerCertificate, s.signer, now)
			check(t, err)
			result, err = reopened.Revoke(ctx, "principal", target, "compromise", "test", now)
			check(t, err)
			if result.Payload.Generation != 2 || len(result.Payload.Revocations) != 1 {
				t.Fatal("retry lost exact revocation", result)
			}
		})
	}
}

func TestSnapshotAuthorityRejectsWrongOrMissingState(t *testing.T) {
	for _, problem := range []string{"missing", "corrupt", "wrong-key", "wrong-role", "wrong-pin", "ca-revocation"} {
		t.Run(problem, func(t *testing.T) {
			s, p := fixture(t)
			snapshot, err := s.Load(now)
			check(t, err)
			switch problem {
			case "missing":
				check(t, os.Remove(filepath.Join(p.Dir(), filename)))
			case "corrupt":
				check(t, p.WriteFile(filename, []byte("{}")))
			case "wrong-key":
				s.signer = key(t)
			case "wrong-role":
				s.signerCertificate = snapshot.Payload.Issuers[0]
			case "wrong-pin":
				other, _ := fixture(t)
				s.anchor = other.anchor
			case "ca-revocation":
				_, id, err := certificates.Parse(s.signerCertificate)
				check(t, err)
				if _, err := s.Revoke(ctx, "certificate", id.Serial, "compromise", "test", now); err == nil {
					t.Fatal("CA compromise treated as leaf revocation")
				}
				return
			}
			if _, err := Open(p, s.anchor, s.signerCertificate, s.signer, now); err == nil {
				t.Fatal("opened invalid authority")
			}
		})
	}
}
