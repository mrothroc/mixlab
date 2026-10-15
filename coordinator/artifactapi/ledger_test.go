package artifactapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

func TestDurableGrantUses(t *testing.T) {
	dir, e := filepath.EvalSymlinks(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	if e = os.Chmod(dir, 0700); e != nil {
		t.Fatal(e)
	}
	p, e := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal, Role: "coordinator"})
	if e != nil {
		t.Fatal(e)
	}
	var g run.Grant
	b, e := os.ReadFile("../../diloco/testdata/grant.json")
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &g); e != nil {
		t.Fatal(e)
	}
	g.MaximumUses = 2
	g.SHA256, _ = g.Digest()
	proof := trust.AcceptedProof{Digest: g.SHA256, Generation: 1}
	use := func(id string) run.GrantUse {
		return run.GrantUse{Format: "mixlab_artifact_use_v1", GrantHash: g.SHA256, GrantID: g.ID, Operation: g.Operation, UploadID: g.UploadID, IdempotencyID: id}
	}
	l, _ := OpenLedger(p)
	ctx := context.Background()
	failed := errors.New("partial transfer")
	if e = l.WithUse(ctx, g, use("a"), proof, func() error { return failed }); !errors.Is(e, failed) {
		t.Fatal(e)
	}
	l, _ = OpenLedger(p)
	var calls atomic.Int32
	var wg sync.WaitGroup
	results := make(chan error, 10)
	for i := range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			other, _ := OpenLedger(p)
			results <- other.WithUse(ctx, g, use(fmt.Sprint("new-", i)), proof, func() error { calls.Add(1); return nil })
		}()
	}
	wg.Wait()
	close(results)
	accepted := 0
	for e := range results {
		if e == nil {
			accepted++
		}
	}
	if accepted != 1 || calls.Load() != 1 {
		t.Fatal("overspent grant", accepted, calls.Load())
	}
	entries, e := os.ReadDir(dir)
	if e != nil || len(entries) != 3 {
		t.Fatal("rejected uses created lock inodes", len(entries), e)
	}
	if e = l.WithUse(ctx, g, use("a"), proof, func() error { return nil }); e != nil {
		t.Fatal("exact restart retry rejected", e)
	}
	changed := g
	changed.PrincipalID = "other"
	changed.SHA256, _ = changed.Digest()
	u := use("a")
	u.GrantHash = changed.SHA256
	if e = l.WithUse(ctx, changed, u, proof, func() error { t.Fatal("conflicting grant reached storage"); return nil }); e == nil {
		t.Fatal("grant ID rebound")
	}
}
