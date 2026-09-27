package localstate

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"errors"
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

var now = time.Date(2026, 9, 25, 12, 0, 0, 0, time.UTC)

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

type fixture struct {
	a       trust.Anchor
	signer  []byte
	key     ed25519.PrivateKey
	payload trust.Snapshot
	path    statehome.Path
	store   *Store
}

func setup(t *testing.T) fixture {
	t.Helper()
	_, rootKey, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	cluster, err := certificates.NewID()
	check(t, err)
	root, err := certificates.CreateRoot(cluster, rootKey, now)
	check(t, err)
	fp, err := trust.RootFingerprint(rootKey.Public())
	check(t, err)
	a, err := trust.PinRoot(root, fp, now)
	check(t, err)
	_, issuerKey, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	issuer, err := certificates.Issue(a, root, rootKey, certificates.Issuer, "", "", issuerKey.Public(), now)
	check(t, err)
	_, key, err := ed25519.GenerateKey(rand.Reader)
	check(t, err)
	signer, err := certificates.Issue(a, root, rootKey, certificates.SnapshotSigner, "", "", key.Public(), now)
	check(t, err)
	endpoints, err := trust.SignAuthorityEndpoints(a, trust.AuthorityEndpoints{Version: trust.EndpointVersion, Cluster: cluster, Audience: "authority", URLs: []string{"https://authority.example"}, IssuedAt: now.Unix(), ExpiresAt: now.Add(24 * time.Hour).Unix()}, rootKey, now)
	check(t, err)
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	s, err := Open(p, fp)
	check(t, err)
	return fixture{a: a, signer: signer, key: key, path: p, store: s, payload: trust.Snapshot{Version: trust.SnapshotVersion, Cluster: cluster, Issuers: [][]byte{issuer}, EligibleRoles: []trust.Role{trust.Authority, trust.Controller, trust.Coordinator}, Endpoints: endpoints}}
}
func (f fixture) snapshot(t *testing.T, gen uint64, at time.Time, revocations []trust.Revocation) trust.SignedSnapshot {
	t.Helper()
	p := f.payload
	p.Generation = gen
	p.IssuedAt = at.Unix()
	p.ExpiresAt = at.Add(trust.SnapshotLifetime).Unix()
	p.Revocations = revocations
	s, err := trust.SignSnapshot(f.a, p, f.signer, f.key, at)
	check(t, err)
	return s
}

func TestInitializeReopenAdvanceAndStaleRecovery(t *testing.T) {
	f := setup(t)
	first := f.snapshot(t, 1, now, nil)
	check(t, f.store.Initialize(f.a, first, now))
	if err := f.store.Initialize(f.a, first, now); !errors.Is(err, statehome.ErrConflict) {
		t.Fatal("reinitialized identity", err)
	}
	reopened, err := Open(f.path, f.a.Fingerprint())
	check(t, err)
	_, v, err := reopened.Load(now)
	check(t, err)
	if v.Generation() != 1 {
		t.Fatal("generation")
	}
	later := now.Add(time.Hour)
	_, v, err = reopened.Load(later)
	check(t, err)
	b, err := v.Bytes()
	check(t, err)
	if _, err := trust.DecodeSnapshot(f.a, b, later); err == nil {
		t.Fatal("stale restored snapshot accepted as fresh")
	}
	check(t, reopened.Advance(f.snapshot(t, 2, later, nil), later))
	if err := reopened.Advance(first, later); err == nil {
		t.Fatal("rollback accepted")
	}
	_, v, err = f.store.Load(later)
	check(t, err)
	if v.Generation() != 2 {
		t.Fatal("stale writer view")
	}
}

func TestPinnedRootSurvivesRestore(t *testing.T) {
	f := setup(t)
	first := f.snapshot(t, 1, now, nil)
	s, err := Open(f.path, f.a.Fingerprint())
	check(t, err)
	check(t, s.Initialize(f.a, first, now))
	reopened, err := Open(f.path, f.a.Fingerprint())
	check(t, err)
	a, _, err := reopened.Load(now)
	check(t, err)
	if a.Fingerprint() != f.a.Fingerprint() || !bytes.Equal(a.DER(), f.a.DER()) {
		t.Fatal("pin lost on restore")
	}
	check(t, reopened.Advance(f.snapshot(t, 2, now, nil), now))
	a, _, err = reopened.Load(now)
	check(t, err)
	if a.Fingerprint() != f.a.Fingerprint() || !bytes.Equal(a.DER(), f.a.DER()) {
		t.Fatal("pin lost on advance")
	}
}

func TestPersistedRevocationsSurviveRestart(t *testing.T) {
	f := setup(t)
	r := trust.Revocation{Kind: "principal", ID: strings.Repeat("ab", 16), Mode: "compromise", Reason: "compromised", FirstGeneration: 1}
	check(t, f.store.Initialize(f.a, f.snapshot(t, 1, now, []trust.Revocation{r}), now))
	later := now.Add(time.Hour)
	s, err := Open(f.path, f.a.Fingerprint())
	check(t, err)
	before, err := f.path.ReadFile(filename)
	check(t, err)
	if err := s.Advance(f.snapshot(t, 2, later, nil), later); err == nil {
		t.Fatal("revocation lost after restart")
	}
	after, err := f.path.ReadFile(filename)
	check(t, err)
	if !bytes.Equal(before, after) {
		t.Fatal("failed update changed state")
	}
	check(t, s.Advance(f.snapshot(t, 2, later, []trust.Revocation{r}), later))
}

func TestLocalTrustRejectsWrongPinCorruptionAndUnsafeFiles(t *testing.T) {
	for _, kind := range []string{"wrong-pin", "root-replaced", "truncated", "oversized", "mode", "symlink", "hardlink", "signature"} {
		t.Run(kind, func(t *testing.T) {
			f := setup(t)
			check(t, f.store.Initialize(f.a, f.snapshot(t, 1, now, nil), now))
			name := filepath.Join(f.path.Dir(), filename)
			s := f.store
			switch kind {
			case "wrong-pin":
				var err error
				s, err = Open(f.path, strings.Repeat("f", 64))
				check(t, err)
			case "root-replaced":
				other := setup(t)
				b, err := f.path.ReadFile(filename)
				check(t, err)
				var r record
				check(t, json.Unmarshal(b, &r))
				r.Root = other.a.DER()
				b, err = json.Marshal(r)
				check(t, err)
				check(t, f.path.WriteFile(filename, b))
			case "truncated":
				check(t, f.path.WriteFile(filename, []byte("{")))
			case "oversized":
				check(t, f.path.WriteFile(filename, make([]byte, maxStateBytes+1)))
			case "mode":
				check(t, os.Chmod(name, 0644))
			case "symlink":
				check(t, os.Rename(name, name+"-real"))
				check(t, os.Symlink(name+"-real", name))
			case "hardlink":
				check(t, os.Link(name, name+"-link"))
			case "signature":
				b, err := f.path.ReadFile(filename)
				check(t, err)
				var r record
				check(t, json.Unmarshal(b, &r))
				var snap trust.SignedSnapshot
				check(t, json.Unmarshal(r.Snapshot, &snap))
				snap.Signature[0] ^= 1
				r.Snapshot, err = json.Marshal(snap)
				check(t, err)
				b, err = json.Marshal(r)
				check(t, err)
				check(t, f.path.WriteFile(filename, b))
			}
			if _, _, err := s.Load(now); err == nil {
				t.Fatal("accepted corrupt/unsafe state")
			}
			if err := s.Advance(f.snapshot(t, 2, now, nil), now); err == nil {
				t.Fatal("silently repaired corrupt state")
			}
		})
	}
}

func TestConcurrentTrustAdvanceHasOneWinner(t *testing.T) {
	f := setup(t)
	check(t, f.store.Initialize(f.a, f.snapshot(t, 1, now, nil), now))
	next := f.snapshot(t, 2, now, nil)
	var wg sync.WaitGroup
	results := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); results <- f.store.Advance(next, now) }()
	}
	wg.Wait()
	close(results)
	wins := 0
	for err := range results {
		if err == nil {
			wins++
		}
	}
	if wins != 1 {
		t.Fatalf("%d winners", wins)
	}
	_, v, err := f.store.Load(now)
	check(t, err)
	if v.Generation() != 2 {
		t.Fatal("wrong committed generation")
	}
}

func TestLocalTrustRejectsFutureOrMissingState(t *testing.T) {
	f := setup(t)
	if _, _, err := f.store.Load(now); !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
	if err := f.store.Advance(f.snapshot(t, 1, now, nil), now); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("advance initialized missing identity", err)
	}
	future := now.Add(time.Hour)
	check(t, f.store.Initialize(f.a, f.snapshot(t, 1, future, nil), future))
	if _, _, err := f.store.Load(now); err == nil {
		t.Fatal("clock rollback accepted future state")
	}
	for _, pin := range []string{"", strings.Repeat("X", 64), strings.ToUpper(f.a.Fingerprint())} {
		if _, err := Open(f.path, pin); err == nil {
			t.Fatal("noncanonical pin")
		}
	}
}
