package clusterservice

import (
	"context"
	"encoding/xml"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

func spec(role string) Spec {
	args := []string{"agent", "-agent-state-dir", "/Users/test/node", "-agent-listen", "127.0.0.1:7445"}
	if role == "authority" {
		args = []string{"authority", "serve", "-cluster-state-dir", "/Users/test/authority"}
	}
	return Spec{role, "/Applications/Mixlab/mixlab-cluster", args}
}

func TestControlReadsRegistrationAfterInstallLock(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(home, 0700); err != nil {
		t.Fatal(err)
	}
	m := Manager{Platform: "darwin", Home: home, UID: 123, Run: func(context.Context, string, ...string) ([]byte, error) {
		return nil, nil
	}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.Install(ctx, spec("agent")); err != nil {
		t.Fatal(err)
	}
	p, _, err := m.Paths("agent")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	err = p.WithProcessLock(ctx, "install.lock", func() error {
		name := filepath.Join(p.Dir(), "service.json")
		if err := os.Rename(name, name+".pending"); err != nil {
			return err
		}
		go func() {
			_, err := m.Control(ctx, "agent", "stop")
			done <- err
		}()
		select {
		case err := <-done:
			t.Errorf("control read an in-flight registration before obtaining lock: %v", err)
			done <- err
		case <-time.After(100 * time.Millisecond):
		}
		return os.Rename(name+".pending", name)
	})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}

func TestRenderServiceBoundaries(t *testing.T) {
	for _, role := range []string{"agent", "authority"} {
		b, err := Render("darwin", "/Users/a & b/service", spec(role))
		if err != nil {
			t.Fatal(err)
		}
		d := xml.NewDecoder(strings.NewReader(string(b)))
		for {
			_, err = d.Token()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		for _, want := range []string{"RunAtLoad", "KeepAlive", "ThrottleInterval", "Aqua", "a &amp; b", "service-run"} {
			if !strings.Contains(string(b), want) {
				t.Fatal(want, string(b))
			}
		}
		for _, bad := range []string{"UserName", "/bin/sh", "LaunchDaemon"} {
			if strings.Contains(string(b), bad) {
				t.Fatal(bad)
			}
		}
		b, err = Render("linux", "/home/a %h $HOME/service", spec(role))
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"%%h", "$$HOME", "RestartSec=30", "KillMode=mixed", "UMask=0077"} {
			if !strings.Contains(string(b), want) {
				t.Fatal(want, string(b))
			}
		}
	}
	for _, s := range []Spec{{Role: "other"}, {Role: "agent", Binary: "relative", Arguments: []string{"agent"}}, {Role: "agent", Binary: "/bin/cluster", Arguments: []string{"agent", "install"}}} {
		if s.Validate() == nil {
			t.Fatal("accepted", s)
		}
	}
}

func TestServiceLifecyclePreservesStateAndForeignUnits(t *testing.T) {
	for _, platform := range []string{"darwin", "linux"} {
		t.Run(platform, func(t *testing.T) {
			home, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err = os.Chmod(home, 0700); err != nil {
				t.Fatal(err)
			}
			loaded := false
			var calls []string
			m := Manager{Platform: platform, Home: home, UID: 123, Run: func(_ context.Context, command string, args ...string) ([]byte, error) {
				call := strings.Join(append([]string{command}, args...), " ")
				calls = append(calls, call)
				if strings.Contains(call, " print ") || strings.Contains(call, " status ") {
					if !loaded {
						return nil, errors.New("not loaded")
					}
					return []byte("running"), nil
				}
				if strings.Contains(call, " bootstrap ") || strings.Contains(call, " enable ") {
					loaded = true
				}
				if strings.Contains(call, " bootout ") || strings.Contains(call, " disable ") {
					loaded = false
				}
				return nil, nil
			}}
			ctx := context.Background()
			s := spec("agent")
			if err = m.Install(ctx, s); err != nil {
				t.Fatal(err)
			}
			if !loaded {
				t.Fatal("not started")
			}
			if err = m.Install(ctx, s); err != nil {
				t.Fatal(err)
			}
			changed := s
			changed.Binary = "/new/cluster"
			if m.Install(ctx, changed) == nil {
				t.Fatal("changed service accepted")
			}
			p, unit, err := m.Paths("agent")
			if err != nil {
				t.Fatal(err)
			}
			log := TailLog{Path: p}
			if _, err = log.Write([]byte("keep diagnostic")); err != nil {
				t.Fatal(err)
			}
			good, err := os.ReadFile(unit)
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(unit, []byte("foreign"), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err = m.Control(ctx, "agent", "uninstall"); err == nil {
				t.Fatal("removed foreign definition")
			}
			if err = os.WriteFile(unit, good, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err = m.Control(ctx, "agent", "uninstall"); err != nil {
				t.Fatal(err)
			}
			if loaded {
				t.Fatal("not stopped")
			}
			if _, err = os.Stat(unit); !os.IsNotExist(err) {
				t.Fatal(err)
			}
			if b, err := p.ReadFile("service.log"); err != nil || string(b) != "keep diagnostic" {
				t.Fatal(string(b), err)
			}
			if strings.Contains(strings.Join(calls, "\n"), "sudo") {
				t.Fatal(calls)
			}
		})
	}
}

func TestServiceRejectsSymlinkAndBoundsLogs(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err = os.Symlink(dir, link); err != nil {
		t.Fatal(err)
	}
	if publishUnit(filepath.Join(link, "a.plist"), []byte("x")) == nil {
		t.Fatal("followed symlink")
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		t.Fatal(err)
	}
	l := TailLog{Path: p}
	if _, err = l.Write([]byte(strings.Repeat("a", 300<<10))); err != nil {
		t.Fatal(err)
	}
	if _, err = l.Write([]byte("last")); err != nil {
		t.Fatal(err)
	}
	b, err := p.ReadFile("service.log")
	if err != nil || len(b) != 256<<10 || !strings.HasSuffix(string(b), "last") {
		t.Fatal(len(b), err)
	}
}

func TestFailedServiceStopPreservesRegistration(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(home, 0700); err != nil {
		t.Fatal(err)
	}
	failStop := false
	m := Manager{Platform: "linux", Home: home, UID: 123, Run: func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if len(args) > 1 && args[1] == "status" {
			return nil, errors.New("inactive")
		}
		if failStop {
			return []byte("access denied"), errors.New("manager inaccessible")
		}
		return nil, nil
	}}
	if err = m.Install(context.Background(), spec("agent")); err != nil {
		t.Fatal(err)
	}
	failStop = true
	if _, err = m.Control(context.Background(), "agent", "uninstall"); err == nil {
		t.Fatal("deleted registration without confirming stop")
	}
	p, unit, err := m.Paths("agent")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Read(p); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(unit); err != nil {
		t.Fatal(err)
	}
}

func TestUninstallRecoversMissingUnit(t *testing.T) {
	for _, platform := range []string{"darwin", "linux"} {
		t.Run(platform, func(t *testing.T) {
			home, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(home, 0700); err != nil {
				t.Fatal(err)
			}
			missing := false
			var unitPath string
			m := Manager{Platform: platform, Home: home, UID: 123, Run: func(_ context.Context, _ string, args ...string) ([]byte, error) {
				if missing {
					name := Label("agent") + ".service"
					switch args[1] {
					case "stop":
						return []byte("Failed to stop " + name + ": Unit " + name + " not loaded."), errors.New("absent")
					case "disable":
						b, err := os.ReadFile(unitPath)
						if err != nil || !strings.Contains(string(b), "WantedBy=default.target") {
							t.Fatal("disable needs restored install metadata", err)
						}
						return nil, nil
					case "daemon-reload":
						if _, err := os.Stat(unitPath); !os.IsNotExist(err) {
							t.Fatal("final reload ran before unit removal", err)
						}
						return nil, nil
					}
					return []byte("Boot-out failed: 3: No such process"), errors.New("absent")
				}
				return nil, nil
			}}
			ctx := context.Background()
			if err := m.Install(ctx, spec("agent")); err != nil {
				t.Fatal(err)
			}
			p, unit, err := m.Paths("agent")
			if err != nil {
				t.Fatal(err)
			}
			unitPath = unit
			if err := os.Remove(unit); err != nil {
				t.Fatal(err)
			}
			missing = true
			if _, err := m.Control(ctx, "agent", "start"); err == nil {
				t.Fatal("start accepted a missing definition")
			}
			if _, err := m.Control(ctx, "agent", "uninstall"); err != nil {
				t.Fatal(err)
			}
			if _, err := Read(p); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("registration retained", err)
			}
		})
	}
}

func TestLinuxUninstallRetriesFinalReloadFailure(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(home, 0700); err != nil {
		t.Fatal(err)
	}
	failReload := false
	var calls []string
	m := Manager{Platform: "linux", Home: home, UID: 123, Run: func(_ context.Context, _ string, args ...string) ([]byte, error) {
		calls = append(calls, args[1])
		if failReload && args[1] == "daemon-reload" {
			return nil, errors.New("reload failed")
		}
		return nil, nil
	}}
	ctx := context.Background()
	if err := m.Install(ctx, spec("agent")); err != nil {
		t.Fatal(err)
	}
	failReload = true
	if _, err := m.Control(ctx, "agent", "uninstall"); err == nil {
		t.Fatal("ignored final reload failure")
	}
	p, unit, err := m.Paths("agent")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Read(p); err != nil {
		t.Fatal("lost recovery registration", err)
	}
	if _, err := os.Stat(unit); !os.IsNotExist(err) {
		t.Fatal("unit must be removed before reload", err)
	}
	failReload = false
	calls = nil
	if _, err := m.Control(ctx, "agent", "uninstall"); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(calls, ","); got != "stop,disable,daemon-reload" {
		t.Fatal("incorrect cleanup order", got)
	}
	if _, err := Read(p); !os.IsNotExist(err) {
		t.Fatal("registration retained after successful reload", err)
	}
}

func TestLinuxStopPreservesEnablementAndDisableFailurePreservesUnit(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(home, 0700); err != nil {
		t.Fatal(err)
	}
	disables := 0
	m := Manager{Platform: "linux", Home: home, UID: 123, Run: func(_ context.Context, _ string, args ...string) ([]byte, error) {
		if args[1] == "disable" {
			disables++
			return nil, errors.New("service manager unavailable")
		}
		return nil, nil
	}}
	ctx := context.Background()
	if err := m.Install(ctx, spec("agent")); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Control(ctx, "agent", "stop"); err != nil || disables != 0 {
		t.Fatal("stop must preserve login startup", err, disables)
	}
	if _, err := m.Control(ctx, "agent", "uninstall"); err == nil || disables != 1 {
		t.Fatal("uninstall ignored disable failure", err, disables)
	}
	p, unit, err := m.Paths("agent")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Read(p); err != nil {
		t.Fatal("lost retryable registration", err)
	}
	if _, err := os.Stat(unit); err != nil {
		t.Fatal("lost retryable unit", err)
	}
}
