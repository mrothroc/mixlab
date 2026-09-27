package statehome

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestResolvePrecedence(t *testing.T) {
	t.Setenv("MIXLAB_STATE_HOME", "/must-not-read-process-environment")
	o := Options{ExactDir: "/exact", Flag: "/flag", Env: "/env", UserHome: "/home/user"}
	c := Context{Kind: Agent, ClusterID: "c", ID: "n"}
	for _, tc := range []struct {
		name, want string
		options    Options
	}{
		{"exact", "/exact", o},
		{"flag", "/flag/agents/c/n", Options{Flag: o.Flag, Env: o.Env, UserHome: o.UserHome}},
		{"env", "/env/agents/c/n", Options{Env: o.Env, UserHome: o.UserHome}},
		{"default", "/home/user/.mixlab/agents/c/n", Options{UserHome: o.UserHome}},
		{"ignore-invalid-lower-priority", "/exact", Options{ExactDir: o.ExactDir, Flag: "../bad", Env: "../bad"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := Resolve(tc.options, c)
			if err != nil {
				t.Fatal(err)
			}
			if p.Dir() != tc.want || p.Kind() != Agent {
				t.Fatalf("got %#v, want %s", p, tc.want)
			}
		})
	}
}

func TestCanonicalPaths(t *testing.T) {
	for _, tc := range []struct {
		c    Context
		want string
	}{
		{Context{Kind: Authority, ClusterID: "c"}, "clusters/c/authority"},
		{Context{Kind: Principal, ClusterID: "c", Role: "opaque-role", ID: "p"}, "principals/c/opaque-role/p"},
		{Context{Kind: Agent, ClusterID: "c", ID: "n"}, "agents/c/n"},
		{Context{Kind: Worker, ClusterID: "c", ID: "w"}, "workers/c/w"},
		{Context{Kind: Enrollment, ID: "random-id"}, "staging/enrollment/random-id"},
	} {
		p, err := Resolve(Options{Flag: "/state"}, tc.c)
		if err != nil {
			t.Fatal(err)
		}
		if p.Dir() != filepath.Join("/state", tc.want) {
			t.Fatal(p.Dir())
		}
	}
}

func TestPureRelativeResolution(t *testing.T) {
	p, err := Resolve(Options{ExactDir: "./literal-$HOME/~/state//", WorkingDir: "/base"}, Context{Kind: Worker})
	if err != nil {
		t.Fatal(err)
	}
	if p.Dir() != "/base/literal-$HOME/~/state" {
		t.Fatal(p.Dir())
	}
	for _, o := range []Options{
		{}, {ExactDir: "relative"}, {ExactDir: "/a/../b"}, {ExactDir: "a", WorkingDir: "/a/../b"},
		{UserHome: "/a/../b"}, {Flag: "/a\x00b"}, {Env: "/a\\b"},
	} {
		if _, err := Resolve(o, Context{Kind: Worker}); !errors.Is(err, ErrUnsafe) {
			t.Fatalf("%#v: %v", o, err)
		}
	}
}

func TestInvalidContext(t *testing.T) {
	for _, c := range []Context{
		{}, {Kind: "other"}, {Kind: Authority, ID: "p"}, {Kind: Agent, Role: "r"},
		{Kind: Enrollment, ClusterID: "c"}, {Kind: Principal, Role: "../admin"},
		{Kind: Worker, ID: "."}, {Kind: Worker, ID: "a/b"}, {Kind: Worker, ID: "a\\b"},
		{Kind: Worker, ID: "a\x00b"}, {Kind: Worker, ID: temporaryPrefix + "reserved"},
	} {
		if _, err := Resolve(Options{ExactDir: "/exact"}, c); !errors.Is(err, ErrUnsafe) {
			t.Fatalf("%#v: %v", c, err)
		}
	}
	if _, err := Resolve(Options{Flag: "/home"}, Context{Kind: Principal, ClusterID: "c"}); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
}

func privateTemp(t *testing.T) string {
	t.Helper()
	// macOS's temporary directory may start with the /var symlink. Production
	// rejects symlink components, so tests explicitly supply the real path.
	path, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0700); err != nil {
		t.Fatal(err)
	}
	return path
}

func resolved(t *testing.T, o Options, c Context) Path {
	t.Helper()
	p, err := Resolve(o, c)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
