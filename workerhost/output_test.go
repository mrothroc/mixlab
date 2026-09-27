package workerhost

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestOutputLookupNeverAllocatesOrRepairs(t *testing.T) {
	s := runtimeStoreFixture(t)
	ctx := context.Background()
	job, attempt := strings.Repeat("a", 32), strings.Repeat("b", 32)
	before, err := os.ReadDir(s.root.Dir())
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Output(ctx, job, attempt); err == nil {
		t.Fatal("unallocated output accepted")
	}
	after, err := os.ReadDir(s.root.Dir())
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatal("read allocated files")
	}
	p, err := s.Attempt(ctx, job, attempt)
	if err != nil {
		t.Fatal(err)
	}
	ref := artifact.Ref{SHA256: strings.Repeat("c", 64), Bytes: 4}
	b, _ := json.Marshal(ref)
	if err := p.CompareAndSwap(workerjob.OutputReceiptFile, nil, b); err != nil {
		t.Fatal(err)
	}
	_, got, err := s.Output(ctx, job, attempt)
	if err != nil || got != ref {
		t.Fatal(got, err)
	}
	if err := os.Remove(filepath.Join(p.Dir(), "runtime-identity")); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Output(ctx, job, attempt); err == nil {
		t.Fatal("missing identity repaired")
	}
	if _, err := os.Stat(filepath.Join(p.Dir(), "runtime-identity")); !os.IsNotExist(err) {
		t.Fatal(err)
	}
}
