package principalrenewal

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

func pathFor(t *testing.T) statehome.Path {
	t.Helper()
	d, e := filepath.EvalSymlinks(t.TempDir())
	if e != nil {
		t.Fatal(e)
	}
	if e := os.Chmod(d, 0700); e != nil {
		t.Fatal(e)
	}
	p, e := statehome.Resolve(statehome.Options{ExactDir: d}, statehome.Context{Kind: statehome.Principal})
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func identityAt(t time.Time) Identity {
	return Identity{strings.Repeat("a", 32), strings.Repeat("b", 64), t, t.Add(30 * 24 * time.Hour)}
}
func TestScheduleRetryRestartAndRenew(t *testing.T) {
	ctx := context.Background()
	start := time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC)
	i := identityAt(start)
	calls := 0
	offline := true
	p := pathFor(t)
	ports := Ports{Identity: func(time.Time) (Identity, error) { return i, nil }, Renew: func(_ context.Context, now time.Time) error {
		calls++
		if offline {
			return errors.New("unavailable")
		}
		i = Identity{i.Principal, strings.Repeat("c", 64), now, now.Add(30 * 24 * time.Hour)}
		return nil
	}}
	s, e := New(p, ports)
	if e != nil {
		t.Fatal(e)
	}
	r, e := s.Tick(ctx, start)
	if e != nil || calls != 0 || r.Outcome != "scheduled" {
		t.Fatal(r, e, calls)
	}
	due := Due(i)
	if due.Before(start.Add(20*24*time.Hour-10*time.Minute)) || due.After(start.Add(20*24*time.Hour-5*time.Minute)) {
		t.Fatal(due)
	}
	r, e = s.Tick(ctx, due)
	if e == nil || calls != 1 || r.Next != due.Add(5*time.Second).Unix() || r.Outcome != "failed" {
		t.Fatal(r, e, calls)
	}
	s, e = New(p, ports)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = s.Tick(ctx, due.Add(time.Second)); e != nil || calls != 1 {
		t.Fatal("restart ignored backoff", e, calls)
	}
	r, e = s.Tick(ctx, time.Unix(r.Next, 0))
	if e == nil || calls != 2 || r.Next != due.Add(15*time.Second).Unix() {
		t.Fatal(r, e, calls)
	}
	offline = false
	r, e = s.Tick(ctx, time.Unix(r.Next, 0))
	if e != nil || calls != 3 || r.Outcome != "renewed" || r.Failures != 0 {
		t.Fatal(r, e, calls)
	}
	if _, e = s.Tick(ctx, i.Issued.Add(time.Minute)); e != nil || calls != 3 {
		t.Fatal("renewal repeated", e, calls)
	}
}
func TestScheduleExpiryCorruptionAndIndependentJitter(t *testing.T) {
	start := time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC)
	i := identityAt(start)
	called := false
	p := pathFor(t)
	s, e := New(p, Ports{Identity: func(time.Time) (Identity, error) { return i, nil }, Renew: func(context.Context, time.Time) error { called = true; return nil }})
	if e != nil {
		t.Fatal(e)
	}
	if _, e = s.Tick(context.Background(), i.Expires); !errors.Is(e, ErrReenrollmentRequired) || called {
		t.Fatal("expired identity attempted recovery", e)
	}
	if e = p.WriteFile(filename, []byte(`{}`)); e != nil {
		t.Fatal(e)
	}
	if _, e = s.Tick(context.Background(), Due(i)); e == nil || called {
		t.Fatal("corrupt schedule recreated")
	}
	j := i
	j.Principal = strings.Repeat("c", 32)
	if Due(i).Equal(Due(j)) || !Due(i).Equal(Due(identityAt(start))) {
		t.Fatal("jitter not independent/deterministic")
	}
}
