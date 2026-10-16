package clusterservice

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestCoordinatorAllowlistRequiresEveryOptionExactlyOnce(t *testing.T) {
	good := coordinatorSpec()
	for i := 2; i < len(good.Arguments); i += 2 {
		for _, duplicate := range []bool{false, true} {
			s := coordinatorSpec()
			if duplicate {
				s.Arguments = append(s.Arguments, s.Arguments[i:i+2]...)
			} else {
				s.Arguments = append(s.Arguments[:i:i], s.Arguments[i+2:]...)
			}
			if s.Validate() == nil {
				t.Fatalf("accepted missing/duplicate option %s: %v", good.Arguments[i], s.Arguments)
			}
		}
	}
	for _, args := range [][]string{{"coordinator"}, {"coordinator", "init"}, {"agent", "serve"}, append(coordinatorSpec().Arguments, "-artifact-dir"), append(coordinatorSpec().Arguments, "-state-home", "/private/other")} {
		s := coordinatorSpec()
		s.Arguments = args
		if s.Validate() == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	// Similar names are independent siblings; sharing a parent is permitted.
	s := coordinatorSpec()
	s.Arguments[3], s.Arguments[5] = "/private/state", "/private/state-other"
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{"/private/state/child", "/private/state", "/private/control\npath", "/private/control\x00path"} {
		s.Arguments[5] = value
		if s.Validate() == nil {
			t.Fatalf("accepted overlapping/unsafe root %q", value)
		}
	}
}

func TestCoordinatorInstallPersistsExactIndependentPaths(t *testing.T) {
	for _, platform := range []string{"darwin", "linux"} {
		t.Run(platform, func(t *testing.T) {
			home, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(home, 0700); err != nil {
				t.Fatal(err)
			}
			var calls []string
			m := Manager{Platform: platform, Home: home, UID: 123, Run: func(_ context.Context, command string, args ...string) ([]byte, error) {
				calls = append(calls, strings.Join(append([]string{command}, args...), " "))
				return nil, nil
			}}
			s := coordinatorSpec()
			ctx := context.Background()
			for i := 0; i < 2; i++ {
				if err := m.Install(ctx, s); err != nil {
					t.Fatal(err)
				}
			}
			p, _, err := m.Paths("coordinator")
			if err != nil {
				t.Fatal(err)
			}
			got, err := Read(p)
			if err != nil || !reflect.DeepEqual(got, s) {
				t.Fatalf("%+v %v", got, err)
			}
			changed := coordinatorSpec()
			changed.Arguments[3] = "/private/new-principal"
			if m.Install(ctx, changed) == nil {
				t.Fatal("silently changed coordinator identity")
			}
			for _, action := range []string{"status", "stop", "start", "uninstall"} {
				if _, err := m.Control(ctx, "coordinator", action); err != nil {
					t.Fatal(action, err)
				}
			}
			if _, err := Read(p); !os.IsNotExist(err) {
				t.Fatal("retained registration", err)
			}
			for _, call := range calls {
				if strings.Contains(call, "sudo") || strings.Contains(call, "cluster-agent") || strings.Contains(call, "cluster-authority") {
					t.Fatal(call)
				}
			}
		})
	}
}

func coordinatorSpec() Spec {
	a := []string{"coordinator", "serve"}
	for _, name := range CoordinatorPathFlags() {
		a = append(a, "-"+name, "/private/mixlab/"+name)
	}
	a = append(a, "-coordinator-listen", "127.0.0.1:7446")
	return Spec{Role: "coordinator", Binary: "/Applications/Mixlab/mixlab-cluster", Arguments: a}
}

func TestCoordinatorServiceContextAllowlist(t *testing.T) {
	good := coordinatorSpec()
	if err := good.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, name := range CoordinatorPathFlags() {
		t.Run(name, func(t *testing.T) {
			for _, value := range []string{"", "relative", "/private/mixlab/../other", "/private/mixlab/principal-state-dir", "/private/mixlab/principal-state-dir/child"} {
				if name == "principal-state-dir" && value == "/private/mixlab/principal-state-dir" {
					continue
				}
				s := coordinatorSpec()
				for i := range s.Arguments {
					if s.Arguments[i] == "-"+name {
						s.Arguments[i+1] = value
					}
				}
				// A principal path moved to its own child does not overlap others.
				if name == "principal-state-dir" && strings.HasSuffix(value, "/child") {
					continue
				}
				if err := s.Validate(); err == nil {
					t.Fatalf("accepted %s=%q", name, value)
				}
			}
		})
	}
	for _, tail := range [][]string{{"-cluster-state-dir", "/authority"}, {"-coordinator-listen", "127.0.0.1:7446"}, {"-shell", "sh"}} {
		s := coordinatorSpec()
		s.Arguments = append(s.Arguments, tail...)
		if s.Validate() == nil {
			t.Fatalf("accepted unexpected options %v", tail)
		}
	}
	for _, listen := range []string{"", ":7446", "host", "host:0", "host:65536", "host:http"} {
		s := coordinatorSpec()
		s.Arguments[len(s.Arguments)-1] = listen
		if s.Validate() == nil {
			t.Fatalf("accepted listener %q", listen)
		}
	}
}

func TestCoordinatorServiceReusesOSLifecycle(t *testing.T) {
	for _, platform := range []string{"darwin", "linux"} {
		s := coordinatorSpec()
		m := Manager{Platform: platform, Home: "/Users/test", UID: 123}
		path, unit, err := m.Paths(s.Role)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(filepath.Base(unit), Label(s.Role)) {
			t.Fatalf("wrong coordinator unit %s", unit)
		}
		b, err := Render(platform, path.Dir(), s)
		if err != nil || !strings.Contains(string(b), "service-run") {
			t.Fatalf("render coordinator: %s %v", b, err)
		}
		if platform == "darwin" && !strings.Contains(string(b), "Aqua") {
			t.Fatal("coordinator changed logged-in session contract")
		}
	}
}
