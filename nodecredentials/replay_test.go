package nodecredentials

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

func fixture(t *testing.T) (statehome.Path, trust.EnvelopeBinding, string) {
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
	id := strings.Repeat("ab", 16)
	b := trust.EnvelopeBinding{Cluster: id, Node: id, Lease: id, Job: id, Run: id, Worker: id, Attempt: id, Audience: "test", CredentialKind: "test_v1", IssuedAt: 100, ExpiresAt: 200}
	return p, b, strings.Repeat("ab", 32)
}

func TestReplayRestartAndConcurrency(t *testing.T) {
	p, b, d := fixture(t)
	if _, err := OpenReplay(p, b, d); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("missing state accepted", err)
	}
	j, err := InitializeReplay(p, b, d)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := InitializeReplay(p, b, d); !errors.Is(err, statehome.ErrConflict) {
		t.Fatal("state replaced", err)
	}
	var wg sync.WaitGroup
	results := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); results <- j.ReserveEnvelope(b, d) }()
	}
	wg.Wait()
	close(results)
	wins := 0
	for err := range results {
		if err == nil {
			wins++
		} else if !errors.Is(err, ErrConsumed) && !errors.Is(err, statehome.ErrConflict) {
			t.Fatal(err)
		}
	}
	if wins != 1 {
		t.Fatal("multiple reservations", wins)
	}
	j, err = OpenReplay(p, b, d)
	if err != nil {
		t.Fatal(err)
	}
	if yes, err := j.Reserved(); err != nil || !yes {
		t.Fatal("reservation lost", err)
	}
	if err := j.ReserveEnvelope(b, d); !errors.Is(err, ErrConsumed) {
		t.Fatal("replay after restart", err)
	}
}

type uncertainStorage struct{ storage }

func (s uncertainStorage) CompareAndSwap(n string, a, b []byte) error {
	if err := s.storage.CompareAndSwap(n, a, b); err != nil {
		return err
	}
	return errors.New("uncertain directory sync")
}
func TestReplayUncertainPublicationFailsClosed(t *testing.T) {
	p, b, d := fixture(t)
	j, err := InitializeReplay(p, b, d)
	if err != nil {
		t.Fatal(err)
	}
	j.path = uncertainStorage{p}
	if err := j.ReserveEnvelope(b, d); err == nil {
		t.Fatal("uncertainty ignored")
	}
	j, err = OpenReplay(p, b, d)
	if err != nil {
		t.Fatal(err)
	}
	if err := j.ReserveEnvelope(b, d); !errors.Is(err, ErrConsumed) {
		t.Fatal("uncertain reservation reset", err)
	}
}
func TestReplayRejectsSubstitutionAndCorruption(t *testing.T) {
	p, b, d := fixture(t)
	j, err := InitializeReplay(p, b, d)
	if err != nil {
		t.Fatal(err)
	}
	other := b
	other.Attempt = strings.Repeat("cd", 16)
	if err := j.ReserveEnvelope(other, d); err == nil {
		t.Fatal("wrong attempt reserved")
	}
	if _, err := OpenReplay(p, other, d); err == nil {
		t.Fatal("substituted binding accepted")
	}
	if _, err := OpenReplay(p, b, strings.Repeat("cd", 32)); err == nil {
		t.Fatal("substituted digest accepted")
	}
	if err := p.WriteFile(filename, []byte("{}")); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenReplay(p, b, d); err == nil {
		t.Fatal("corrupt state accepted")
	}
	if err := j.ReserveEnvelope(b, d); err == nil {
		t.Fatal("corrupt state repaired")
	}
}
