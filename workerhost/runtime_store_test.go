package workerhost

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
)

func runtimeStoreFixture(t *testing.T) *RuntimeStore {
	t.Helper()
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(base, "runtime")}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	s, err := InitializeRuntimeStore(p)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRuntimeStoreNeverRecreatesPublishedAttempt(t *testing.T) {
	for _, missing := range []string{"directory", "identity", "index"} {
		t.Run(missing, func(t *testing.T) {
			s := runtimeStoreFixture(t)
			job, attempt := strings.Repeat("a", 32), strings.Repeat("b", 32)
			p, err := s.Attempt(context.Background(), job, attempt)
			if err != nil {
				t.Fatal(err)
			}
			reopened, err := OpenRuntimeStore(s.root)
			if err != nil {
				t.Fatal(err)
			}
			again, err := reopened.Attempt(context.Background(), job, attempt)
			if err != nil || again.Dir() != p.Dir() {
				t.Fatal("allocation retry", err)
			}
			switch missing {
			case "directory":
				err = os.RemoveAll(p.Dir())
			case "identity":
				err = os.Remove(filepath.Join(p.Dir(), "runtime-identity"))
			case "index":
				err = os.Remove(filepath.Join(s.root.Dir(), runtimeIndexFile))
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.Attempt(context.Background(), job, attempt); err == nil {
				t.Fatal("recreated lost runtime evidence")
			}
			if _, err := InitializeRuntimeStore(s.root); err == nil {
				t.Fatal("reinitialized published runtime root")
			}
		})
	}
}

func TestRuntimeStoreInterruptedAllocationAndValidation(t *testing.T) {
	s := runtimeStoreFixture(t)
	job, attempt := strings.Repeat("a", 32), strings.Repeat("b", 32)
	index := runtimeIndex{runtimeIndexVersion, []runtimeEntry{{Job: job, Attempt: attempt}}}
	b, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.root.WriteFile(runtimeIndexFile, b); err != nil {
		t.Fatal(err)
	}
	p, err := s.Attempt(context.Background(), job, attempt)
	if err != nil {
		t.Fatal(err)
	}
	// Simulate child publication followed by a crash before the root ready write.
	if err := s.root.WriteFile(runtimeIndexFile, b); err != nil {
		t.Fatal(err)
	}
	again, err := s.Attempt(context.Background(), job, attempt)
	if err != nil || again.Dir() != p.Dir() {
		t.Fatal("pending allocation not recovered", err)
	}
	for _, id := range []string{"", "../escape", strings.Repeat("A", 32)} {
		if _, err := s.Attempt(context.Background(), id, attempt); err == nil {
			t.Fatal("accepted invalid identifier")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Attempt(ctx, job, attempt); err == nil {
		t.Fatal("ignored cancellation")
	}
}
