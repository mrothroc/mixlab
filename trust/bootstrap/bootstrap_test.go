package bootstrap

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

var testNow = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
var testContext = context.Background()

func require(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func fixture(t *testing.T) Config {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	require(t, err)
	require(t, os.Chmod(dir, 0700))
	ids, err := NewIDs()
	require(t, err)
	p, err := resolved(filepath.Join(dir, "ca"), statehome.Authority)
	require(t, err)
	c := Config{Cluster: ids.Cluster, Authority: p, Backend: "file", Endpoint: "https://localhost:7443", Audience: "mixlab-trust"}
	for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
		final, err := resolved(filepath.Join(dir, string(role)), statehome.Principal)
		require(t, err)
		stage, err := resolved(filepath.Join(dir, "stage-"+string(role)), statehome.Enrollment)
		require(t, err)
		c.Principals = append(c.Principals, Target{role, []string{ids.Authority, ids.Controller, ids.Coordinator}[i], final, stage})
	}
	return c
}

func readRunner(t *testing.T, c Config) *runner {
	t.Helper()
	x := &runner{path: c.Authority, store: c.Authority, now: testNow}
	require(t, x.read())
	return x
}

func TestInitializeSeparatedIdentitiesAndRecovery(t *testing.T) {
	c := fixture(t)
	r, err := Initialize(testContext, c, testNow)
	require(t, err)
	if r.Cluster != c.Cluster || len(r.Principals) != 3 {
		t.Fatal(r)
	}
	x := readRunner(t, c)
	if !x.r.Ready {
		t.Fatal("not ready")
	}
	seen := map[string]bool{}
	for n, v := range x.r.Contexts {
		p, err := selectedPath(v)
		require(t, err)
		files, err := filepath.Glob(filepath.Join(p.Dir(), "key-*.json"))
		require(t, err)
		// Includes key-context and key-intent files; private key files have opaque IDs.
		private := 0
		for _, f := range files {
			if len(filepath.Base(f)) == len("key-")+32+len(".json") {
				private++
			}
		}
		if private != len(v.Keys) {
			t.Fatalf("%s private keys=%d want=%d", v.Owner, private, len(v.Keys))
		}
		for _, k := range v.Keys {
			if seen[string(k.Handle.PublicKey)] {
				t.Fatal("reused key")
			}
			seen[string(k.Handle.PublicKey)] = true
		}
		if n > 0 {
			require(t, x.verifyPrincipal(p, n))
			b, err := p.ReadFileLimit(credentialFile, maxBytes)
			require(t, err)
			for _, k := range x.r.Contexts[0].Keys {
				if bytes.Contains(b, []byte(k.Handle.ID)) {
					t.Fatal("CA private handle leaked to principal")
				}
			}
			if _, err := os.Lstat(v.Staging); !os.IsNotExist(err) {
				t.Fatal("staging remains", err)
			}
		}
	}
	if len(seen) != 6 {
		t.Fatal(len(seen))
	}
	before := bytes.Clone(x.raw)
	// Expired initial snapshots may be restored as history, never refreshed by init.
	r2, err := Recover(testContext, c.Authority, testNow.Add(time.Hour))
	require(t, err)
	if !reflect.DeepEqual(r, r2) {
		t.Fatal("recovery changed identity")
	}
	if !bytes.Equal(before, readRunner(t, c).raw) {
		t.Fatal("recovery mutated ready state")
	}
	a, err := x.anchor()
	require(t, err)
	if _, err := trust.VerifySnapshot(a, *x.r.Snapshot, testNow.Add(time.Hour)); err == nil {
		t.Fatal("stale snapshot authorized")
	}
	if _, err := Initialize(testContext, c, testNow); err == nil {
		t.Fatal("init replaced identity")
	}
}

func TestInitializeConcurrent(t *testing.T) {
	c := fixture(t)
	var wg sync.WaitGroup
	results := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); _, err := Initialize(testContext, c, testNow); results <- err }()
	}
	wg.Wait()
	close(results)
	success := 0
	for err := range results {
		if err == nil {
			success++
		}
	}
	if success != 1 {
		t.Fatalf("successful initializations=%d", success)
	}
	_, err := Recover(testContext, c.Authority, testNow)
	require(t, err)
}

func TestRecoveryNeverReplacesMissingOrChangedIdentity(t *testing.T) {
	for _, problem := range []string{"missing-key", "missing-final", "changed-marker", "changed-principal", "backend", "noncanonical"} {
		t.Run(problem, func(t *testing.T) {
			c := fixture(t)
			_, err := Initialize(testContext, c, testNow)
			require(t, err)
			x := readRunner(t, c)
			p := c.Principals[1].Final
			switch problem {
			case "missing-key":
				require(t, os.Remove(filepath.Join(p.Dir(), "key-"+x.r.Contexts[2].Keys[0].Handle.ID+".json")))
			case "missing-final":
				require(t, os.Rename(p.Dir(), p.Dir()+"-held"))
			case "changed-marker":
				require(t, p.WriteFile(contextFile, []byte("{}")))
			case "changed-principal":
				require(t, p.WriteFile(credentialFile, []byte("{}")))
			case "backend":
				x.r.Backend = "keychain"
				b, err := encode(x.r)
				require(t, err)
				require(t, c.Authority.WriteFile(stateFile, b))
			case "noncanonical":
				require(t, c.Authority.WriteFile(stateFile, append(x.raw, '\n')))
			}
			before, err := c.Authority.ReadFileLimit(stateFile, maxBytes)
			require(t, err)
			if _, err := Recover(testContext, c.Authority, testNow); err == nil {
				t.Fatal("recovered changed identity")
			}
			after, err := c.Authority.ReadFileLimit(stateFile, maxBytes)
			require(t, err)
			if !bytes.Equal(before, after) {
				t.Fatal("recovery altered identity")
			}
		})
	}
}

var errInjected = errors.New("injected publication failure")

type faultJournal struct {
	statehome.Path
	at, calls int
	after     bool
}

func (s *faultJournal) CompareAndSwap(name string, old, next []byte) error {
	s.calls++
	fail := s.calls == s.at
	if fail && !s.after {
		return errInjected
	}
	if err := s.Path.CompareAndSwap(name, old, next); err != nil {
		return err
	}
	if fail {
		return errInjected
	}
	return nil
}

func TestRecoveryAcrossEveryBootstrapPublication(t *testing.T) {
	// Walk every journal boundary, including failures after a commit whose
	// outcome the caller did not receive. Missing key intents fail closed.
	for _, after := range []bool{false, true} {
		for at := 1; at < 50; at++ {
			c := fixture(t)
			require(t, prepare(testContext, c, testNow))
			x := readRunner(t, c)
			fault := &faultJournal{Path: c.Authority, at: at, after: after}
			x.store = fault
			err := x.run(testContext)
			if err == nil {
				if fault.calls >= at {
					t.Fatal("fault not reached")
				}
				break
			}
			if !errors.Is(err, errInjected) {
				t.Fatalf("at=%d after=%v: %v", at, after, err)
			}
			saved := readRunner(t, c)
			missingIntent := false
			for _, v := range saved.r.Contexts {
				p, e := selectedPath(v)
				require(t, e)
				for _, k := range v.Keys {
					if k.Started && k.Handle == nil {
						_, e := p.ReadFileLimit("key-intent-"+string(k.Slot)+".json", 4096)
						if errors.Is(e, os.ErrNotExist) {
							missingIntent = true
						}
					}
				}
			}
			_, err = Recover(testContext, c.Authority, testNow)
			if missingIntent {
				if err == nil {
					t.Fatal("silently regenerated missing intent")
				}
			} else {
				require(t, err)
			}
			recovered := readRunner(t, c)
			for n, v := range saved.r.Contexts {
				for k, old := range v.Keys {
					got := recovered.r.Contexts[n].Keys[k]
					if old.Handle != nil && !reflect.DeepEqual(old.Handle, got.Handle) {
						t.Fatal("recovery changed protected handle")
					}
					if len(old.Certificate) > 0 && !bytes.Equal(old.Certificate, got.Certificate) {
						t.Fatal("recovery reissued committed certificate")
					}
				}
			}
		}
	}
}

func TestInvalidPlanHasNoSideEffects(t *testing.T) {
	for _, problem := range []string{"overlap", "role", "endpoint", "backend", "duplicate-id"} {
		t.Run(problem, func(t *testing.T) {
			c := fixture(t)
			switch problem {
			case "overlap":
				c.Principals[0].Staging = c.Principals[1].Staging
			case "role":
				c.Principals[0].Role = trust.Node
			case "endpoint":
				c.Endpoint = "http://localhost:7443"
			case "backend":
				c.Backend = "unknown"
			case "duplicate-id":
				c.Principals[0].Principal = c.Principals[1].Principal
			}
			if _, err := Initialize(testContext, c, testNow); err == nil {
				t.Fatal("accepted invalid plan")
			}
			if _, err := os.Lstat(c.Authority.Dir()); !os.IsNotExist(err) {
				t.Fatal("published invalid plan", err)
			}
		})
	}
}
