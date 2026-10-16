package admission

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
)

func inboxFixture(t *testing.T) (*Inbox, Submission) {
	t.Helper()
	x := setup(t)
	s, err := x.s.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	i, err := OpenInbox(p)
	if err != nil {
		t.Fatal(err)
	}
	s.Request.Run = strings.Repeat("a", 32)
	return i, Submission{Owner: strings.Repeat("d", 32), Launch: s.Request, Specification: s.Specification, Deadline: s.Deadline}
}

func TestServiceAcceptanceSurvivesClientAndServiceExit(t *testing.T) {
	i, q := inboxFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	if err := i.Accept(ctx, q); err != nil {
		t.Fatal(err)
	}
	cancel()
	restarted, err := OpenInbox(i.path)
	if err != nil {
		t.Fatal(err)
	}
	ctx = context.Background()
	ids, err := restarted.IDs(ctx)
	if err != nil || len(ids) != 1 || ids[0] != q.Launch.Run {
		t.Fatalf("lost accepted run: %v %v", ids, err)
	}
	s, err := restarted.Resume(ctx, ids[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := restarted.Accept(ctx, q); err != nil {
		t.Fatal("lost ack retry failed", err)
	}
	if err := s.Cancel(ctx); err != nil {
		t.Fatal(err)
	}
	s, err = restarted.Resume(ctx, ids[0])
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := s.Snapshot(ctx)
	if err != nil || !snapshot.Aborting {
		t.Fatal("restart discarded precommit cancellation", err)
	}
}

func TestServiceAcceptanceRejectsIDReuseAndMissingState(t *testing.T) {
	i, q := inboxFixture(t)
	ctx := context.Background()
	if err := i.Accept(ctx, q); err != nil {
		t.Fatal(err)
	}
	changed := q
	changed.Owner = strings.Repeat("e", 32)
	if err := i.Accept(ctx, changed); err == nil {
		t.Fatal("another owner reused accepted run")
	}
	for _, id := range []string{"../other", strings.ToUpper(q.Launch.Run), "short"} {
		if id == q.Launch.Run {
			continue
		}
		if _, err := i.Read(ctx, id); err == nil {
			t.Fatal("accepted invalid run ID", id)
		}
	}
	s, err := i.Resume(ctx, q.Launch.Run)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(s.path.Dir(), file)); err != nil {
		t.Fatal(err)
	}
	if _, err := i.Resume(ctx, q.Launch.Run); err == nil {
		t.Fatal("recreated a missing saga")
	}
}

func TestServiceInboxRejectsCorruptAcceptedInput(t *testing.T) {
	i, q := inboxFixture(t)
	ctx := context.Background()
	if err := i.Accept(ctx, q); err != nil {
		t.Fatal(err)
	}
	q.Launch.Run = strings.Repeat("f", 32)
	b, err := json.Marshal(q)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := i.IDs(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := i.path.WriteFile(ids[0]+".submission", b); err != nil {
		t.Fatal(err)
	}
	if _, err := i.IDs(ctx); err == nil {
		t.Fatal("accepted mismatched durable run identity")
	}
}
