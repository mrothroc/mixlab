package clusterapp

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/enrollment/enrollee"
	"github.com/mrothroc/mixlab/trust/principal"
)

func enrolleePaths(t *testing.T, ca statehome.Path) (statehome.Path, statehome.Path) {
	t.Helper()
	root := filepath.Dir(ca.Dir())
	stage, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "new-node-staging")}, statehome.Context{Kind: statehome.Enrollment})
	check(t, err)
	final, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "new-node")}, statehome.Context{Kind: statehome.Principal})
	check(t, err)
	return stage, final
}

func TestEnrolleeProtectedPublication(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	v, err := a.Current(ctx, testNow)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, v, testNow)
	check(t, err)
	defer i.Clear()
	stage, final := enrolleePaths(t, p)
	c, err := enrollee.Prepare(ctx, stage, final, a.Anchor, trust.Node, "file", testNow)
	check(t, err)
	defer func() { _ = c.Close() }()
	k, x, err := c.Keys()
	check(t, err)
	q, err := enrollment.NewRequest(i, k, x, testNow)
	check(t, err)
	r, err := a.Enrollment.Consume(ctx, q, i.Secret, v, testNow)
	check(t, err)
	check(t, c.CompleteProvisioned(ctx, i, q, r, testNow))
	if _, err := os.Stat(stage.Dir()); !os.IsNotExist(err) {
		t.Fatal("staging not promoted", err)
	}
	installed, err := principal.Open(final, testNow)
	check(t, err)
	defer func() { _ = installed.Close() }()
	s, _, err := installed.Active(testNow)
	check(t, err)
	if s.Principal != r.Approval.Principal || s.Role != trust.Node || s.EnvelopeKey == nil {
		t.Fatal("installed identity mismatch")
	}
	if err := c.Abort(ctx); err == nil {
		t.Fatal("published identity destroyed")
	}
	if other, err := enrollee.Prepare(ctx, stage, final, a.Anchor, trust.Node, "file", testNow); err == nil {
		_ = other.Close()
		t.Fatal("existing identity replaced")
	}
}

func TestEnrolleeAbortRemovesProvisionalKeys(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	defer func() { _ = a.Close() }()
	stage, final := enrolleePaths(t, p)
	c, err := enrollee.Prepare(ctx, stage, final, a.Anchor, trust.Node, "file", testNow)
	check(t, err)
	check(t, c.Abort(ctx))
	if _, err := os.Stat(stage.Dir()); !os.IsNotExist(err) {
		t.Fatal("provisional context retained", err)
	}
	if _, err := os.Stat(final.Dir()); !os.IsNotExist(err) {
		t.Fatal("abort created final identity", err)
	}
}
