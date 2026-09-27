package keylifecycle

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
)

var ctx = context.Background()

func setup(t *testing.T, owner string) (*Manager, statehome.Path, *securekeys.Store) {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	k, err := securekeys.OpenSelected("file", p, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
	if err != nil {
		t.Fatal(err)
	}
	m, err := Initialize(ctx, p, k, owner)
	if err != nil {
		t.Fatal(err)
	}
	return m, p, k
}
func require(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestCreationRotationRetirement(t *testing.T) {
	m, p, k := setup(t, "node")
	r, err := m.Create(ctx, Principal)
	require(t, err)
	old := *r.Active
	if r.Stage != "active" || r.Generation != 1 {
		t.Fatal(r)
	}
	if _, err := m.Create(ctx, Principal); !errors.Is(err, statehome.ErrExists) {
		t.Fatal("recreated active key", err)
	}
	m, err = Open(p, k, "node")
	require(t, err)
	r, err = m.Recover(ctx, Principal)
	require(t, err)
	if r.Active.ID != old.ID {
		t.Fatal("recovery changed identity")
	}
	r, err = m.Rotate(ctx, Principal)
	require(t, err)
	newKey := *r.Candidate
	if r.Stage != "rotation-ready" || r.Active.ID != old.ID || newKey.ID == old.ID || bytes.Equal(newKey.PublicKey, old.PublicKey) {
		t.Fatal("rotation overwrote active", r)
	}
	if _, err := m.Activate(ctx, Principal, "wrong"); !errors.Is(err, statehome.ErrConflict) {
		t.Fatal(err)
	}
	r, err = m.Activate(ctx, Principal, newKey.ID)
	require(t, err)
	if r.Active.ID != newKey.ID || r.Retiring.ID != old.ID {
		t.Fatal(r)
	}
	require(t, k.Inspect(old))
	r, err = m.Retire(ctx, Principal, old.ID)
	require(t, err)
	if r.Stage != "active" || r.Retiring != nil {
		t.Fatal(r)
	}
	if err := k.Inspect(old); !errors.Is(err, securekeys.ErrMissing) {
		t.Fatal("old key retained", err)
	}
	require(t, k.Inspect(newKey))
	r, err = m.Create(ctx, NodeEnvelope)
	require(t, err)
	if r.Active.Version != securekeys.EnvelopeVersion {
		t.Fatal("wrong envelope profile")
	}
	if _, err := m.Create(ctx, Root); err == nil {
		t.Fatal("node created CA")
	}
	if _, err := m.Rotate(ctx, Root); !errors.Is(err, ErrRekeyRequired) {
		t.Fatal("CA implicitly rotated")
	}
}

// A journal can fail before publication or report an error after publication.
// The real underlying CAS still owns durable bytes and process synchronization.
type failingJournal struct {
	journal
	calls, at int
	after     bool
}

func (f *failingJournal) CompareAndSwap(n string, old, b []byte) error {
	f.calls++
	if f.calls == f.at && !f.after {
		return errors.New("injected pre-publication failure")
	}
	if err := f.journal.CompareAndSwap(n, old, b); err != nil {
		return err
	}
	if f.calls == f.at {
		return errors.New("injected post-publication uncertainty")
	}
	return nil
}

func TestCreateCrashRecoveryBoundaries(t *testing.T) {
	for _, at := range []int{1, 2} {
		for _, after := range []bool{false, true} {
			t.Run(string(rune('0'+at))+map[bool]string{true: "after", false: "before"}[after], func(t *testing.T) {
				m, p, k := setup(t, "node")
				m.journal = &failingJournal{journal: p, at: at, after: after}
				if _, err := m.Create(ctx, NodeEnvelope); err == nil {
					t.Fatal("missing injected failure")
				}
				m, err := Open(p, k, "node")
				require(t, err)
				r, err := m.View(NodeEnvelope)
				if at == 1 && !after {
					if !errors.Is(err, os.ErrNotExist) {
						t.Fatal("intent should be absent", err)
					}
				} else {
					require(t, err)
					if at == 1 {
						if _, err := m.Recover(ctx, NodeEnvelope); !errors.Is(err, ErrRecoveryRequired) {
							t.Fatal("regenerated missing candidate", err)
						}
						id := r.Candidate.ID
						r, err = m.Abort(ctx, NodeEnvelope, id)
						require(t, err)
						if r.Stage != "aborted" {
							t.Fatal(r)
						}
						r, err = m.Create(ctx, NodeEnvelope)
						require(t, err)
						if r.Active.ID == id || r.Generation != 2 {
							t.Fatal("reused lost intent")
						}
					} else {
						var id string
						if r.Active != nil {
							id = r.Active.ID
						} else {
							id = r.Candidate.ID
						}
						r, err = m.Recover(ctx, NodeEnvelope)
						require(t, err)
						if r.Stage != "active" || r.Active.ID != id {
							t.Fatal("did not reconcile exact published key")
						}
					}
				}
				entries, err := os.ReadDir(p.Dir())
				require(t, err)
				count := 0
				for _, e := range entries {
					if len(e.Name()) > 4 && e.Name()[:4] == "key-" && len(e.Name()) == 41 {
						count++
					}
				}
				want := 1
				if at == 1 && !after {
					want = 0
				}
				if count != want {
					t.Fatalf("orphan count=%d want=%d", count, want)
				}
			})
		}
	}
}

func TestAbortAndRetireCrashRecovery(t *testing.T) {
	for _, action := range []string{"abort", "retire"} {
		for _, at := range []int{1, 2} {
			for _, after := range []bool{false, true} {
				t.Run(action+string(rune('0'+at))+map[bool]string{true: "after", false: "before"}[after], func(t *testing.T) {
					m, p, k := setup(t, "controller")
					_, err := m.Create(ctx, Principal)
					require(t, err)
					r, err := m.Rotate(ctx, Principal)
					require(t, err)
					victim := r.Candidate.ID
					if action == "retire" {
						r, err = m.Activate(ctx, Principal, victim)
						require(t, err)
						victim = r.Retiring.ID
					}
					keep := r.Active.ID
					m.journal = &failingJournal{journal: p, at: at, after: after}
					if action == "abort" {
						_, err = m.Abort(ctx, Principal, victim)
					} else {
						_, err = m.Retire(ctx, Principal, victim)
					}
					if err == nil {
						t.Fatal("expected injected failure")
					}
					m, err = Open(p, k, "controller")
					require(t, err)
					r, err = m.Recover(ctx, Principal)
					require(t, err)
					if at == 1 && !after {
						if action == "abort" {
							r, err = m.Abort(ctx, Principal, victim)
						} else {
							r, err = m.Retire(ctx, Principal, victim)
						}
						require(t, err)
					}
					if r.Stage != "active" || r.Active.ID != keep {
						t.Fatal("cleanup changed active key", r)
					}
				})
			}
		}
	}
}

func TestConcurrentCreateAndMissingActive(t *testing.T) {
	m, _, k := setup(t, "controller")
	var wg sync.WaitGroup
	results := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, err := m.Create(ctx, Principal); results <- err }()
	}
	wg.Wait()
	close(results)
	success := 0
	for err := range results {
		if err == nil {
			success++
		} else if !errors.Is(err, statehome.ErrExists) {
			t.Fatal(err)
		}
	}
	if success != 1 {
		t.Fatal("duplicate creation", success)
	}
	r, err := m.View(Principal)
	require(t, err)
	require(t, k.Delete(*r.Active))
	if _, err := m.Recover(ctx, Principal); !errors.Is(err, securekeys.ErrMissing) {
		t.Fatal("missing key silently recovered", err)
	}
	if _, err := m.Rotate(ctx, Principal); !errors.Is(err, securekeys.ErrMissing) {
		t.Fatal("missing key silently replaced", err)
	}
}

func TestIntentRejectsCorruptionAndContextChange(t *testing.T) {
	for _, change := range []string{"owner", "backend", "scope", "stage", "profile", "duplicate-field"} {
		t.Run(change, func(t *testing.T) {
			m, p, _ := setup(t, "node")
			r, err := m.Create(ctx, Principal)
			require(t, err)
			switch change {
			case "owner":
				r.Owner = "controller"
			case "backend":
				r.Backend = "keychain"
			case "scope":
				r.Scope = "bad"
			case "stage":
				r.Stage = "creating"
			case "profile":
				r.Active.Version = securekeys.EnvelopeVersion
			}
			b, err := json.Marshal(r)
			require(t, err)
			if change == "duplicate-field" {
				b = append(b[:len(b)-1], []byte(",\"stage\":\"active\"}")...)
			}
			require(t, p.WriteFile(name(Principal), b))
			if _, err := m.Recover(ctx, Principal); err == nil {
				t.Fatal("corrupt intent accepted")
			}
		})
	}
}

func TestOwnerRecordCannotBeReinterpretedOrRecreated(t *testing.T) {
	m, p, k := setup(t, "node")
	if _, err := Open(p, k, "cluster-authority"); err == nil {
		t.Fatal("node context reinterpreted as CA owner")
	}
	other, err := securekeys.OpenFile(p, "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789")
	require(t, err)
	if _, err := Open(p, other, "node"); err == nil {
		t.Fatal("scope changed")
	}
	if _, err := Initialize(ctx, p, k, "node"); !errors.Is(err, statehome.ErrExists) {
		t.Fatal("context reset", err)
	}
	_, err = m.Create(ctx, Principal)
	require(t, err)
	b, err := p.ReadFile(contextName)
	require(t, err)
	require(t, p.CompareAndSwap(contextName, b, nil))
	if _, err := m.Create(ctx, NodeEnvelope); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatal("missing context treated as missing slot", err)
	}
	if _, err := Open(p, k, "node"); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatal("missing context opened", err)
	}
	if _, err := Initialize(ctx, p, k, "node"); !errors.Is(err, ErrRecoveryRequired) {
		t.Fatal("orphan intents silently adopted", err)
	}
}
