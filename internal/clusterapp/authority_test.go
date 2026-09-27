package clusterapp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
)

var testNow = time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
func initialized(t *testing.T) statehome.Path {
	t.Helper()
	return initializedAt(t, "https://authority.example:7443")
}
func initializedAt(t *testing.T, endpoint string) statehome.Path {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	check(t, err)
	check(t, os.Chmod(dir, 0700))
	resolve := func(name string, kind statehome.Kind) statehome.Path {
		p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(dir, name)}, statehome.Context{Kind: kind})
		check(t, err)
		return p
	}
	ids, err := bootstrap.NewIDs()
	check(t, err)
	p := resolve("ca", statehome.Authority)
	c := bootstrap.Config{Cluster: ids.Cluster, Authority: p, Backend: "file", Endpoint: endpoint, Audience: "authority"}
	for i, role := range []trust.Role{trust.Authority, trust.Controller, trust.Coordinator} {
		c.Principals = append(c.Principals, bootstrap.Target{Role: role, Principal: []string{ids.Authority, ids.Controller, ids.Coordinator}[i], Final: resolve(string(role), statehome.Principal), Staging: resolve("stage-"+string(role), statehome.Enrollment)})
	}
	_, err = bootstrap.Initialize(context.Background(), c, testNow)
	check(t, err)
	return p
}

func TestAuthorityRuntimeExplicitInitializationAndRecovery(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	if a, err := OpenAuthority(ctx, p, testNow); err == nil {
		_ = a.Close()
		t.Fatal("implicitly initialized runtime")
	}
	check(t, InitializeAuthority(ctx, p, testNow))
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	v, err := a.Current(ctx, testNow)
	check(t, err)
	i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, "https://authority.example:7443", "authority", time.Minute, v, testNow)
	check(t, err)
	i.Clear()
	later := testNow.Add(time.Hour)
	current, err := a.Current(ctx, later)
	check(t, err)
	if current.Generation() <= v.Generation() {
		t.Fatal("stale snapshot not refreshed")
	}
	check(t, a.Close())
	// Init recovery must not restore the generation-1 principal snapshot.
	_, err = bootstrap.Recover(ctx, p, later)
	check(t, err)
	check(t, InitializeAuthority(ctx, p, later))
	a, err = OpenAuthority(ctx, p, later)
	check(t, err)
	defer func() { _ = a.Close() }()
	got, err := a.Current(ctx, later)
	check(t, err)
	if got.Generation() != current.Generation() {
		t.Fatal("recovery reset authority history")
	}
}

func TestAuthorityRuntimeDoesNotRecreateMissingJournals(t *testing.T) {
	for _, file := range []string{"authority-snapshots.json", "provisioned-enrollment.json", "workload-issuance.json", "workload-issuance.claim", "workload-issuance.ready"} {
		t.Run(file, func(t *testing.T) {
			ctx := context.Background()
			p := initialized(t)
			check(t, InitializeAuthority(ctx, p, testNow))
			check(t, os.Remove(filepath.Join(p.Dir(), file)))
			if a, err := OpenAuthority(ctx, p, testNow); err == nil {
				_ = a.Close()
				t.Fatal("opened missing history")
			}
			if err := InitializeAuthority(ctx, p, testNow); err == nil {
				t.Fatal("explicit recover reset published history")
			}
			if _, err := os.Stat(filepath.Join(p.Dir(), file)); !os.IsNotExist(err) {
				t.Fatal("missing history recreated", err)
			}
		})
	}
}

func TestAuthorityRuntimeRejectsCorruptMarker(t *testing.T) {
	p := initialized(t)
	check(t, p.WriteFile(runtimeFile, []byte(`{}`)))
	if err := InitializeAuthority(context.Background(), p, testNow); err == nil {
		t.Fatal("replaced corrupt marker")
	}
}

func TestAuthorityRuntimeDelayedFirstStart(t *testing.T) {
	p := initialized(t)
	at := testNow.Add(time.Hour)
	check(t, InitializeAuthority(context.Background(), p, at))
	a, err := OpenAuthority(context.Background(), p, at)
	check(t, err)
	defer func() { _ = a.Close() }()
	_, err = a.Current(context.Background(), at)
	check(t, err)
}

func TestAuthorityWorkloadJournalUpgradeRequiresExplicitRecovery(t *testing.T) {
	ctx := context.Background()
	p := initialized(t)
	check(t, InitializeAuthority(ctx, p, testNow))
	snapshots, err := p.ReadFileLimit("authority-snapshots.json", 1<<20)
	check(t, err)
	// Emulate the earlier internal checkpoint, before workload issuance existed.
	for _, file := range []string{"workload-issuance.json", "workload-issuance.claim", "workload-issuance.ready"} {
		check(t, os.Remove(filepath.Join(p.Dir(), file)))
	}
	b, err := p.ReadFileLimit(runtimeFile, 4096)
	check(t, err)
	var r runtimeState
	check(t, json.Unmarshal(b, &r))
	r.Workloads = false
	b, err = json.Marshal(r)
	check(t, err)
	check(t, p.WriteFile(runtimeFile, b))
	if a, err := OpenAuthority(ctx, p, testNow); err == nil {
		_ = a.Close()
		t.Fatal("runtime implicitly upgraded")
	}
	check(t, InitializeAuthority(ctx, p, testNow))
	a, err := OpenAuthority(ctx, p, testNow)
	check(t, err)
	check(t, a.Close())
	after, err := p.ReadFileLimit("authority-snapshots.json", 1<<20)
	check(t, err)
	if string(after) != string(snapshots) {
		t.Fatal("workload initialization reset trust history")
	}
}
