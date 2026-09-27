//go:build darwin || linux

package statehome

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestEnsureContextIsolation(t *testing.T) {
	base := privateTemp(t)
	o := Options{Flag: filepath.Join(base, "home")}
	p := resolved(t, o, Context{Kind: Agent, ClusterID: "c", ID: "n"})
	if _, err := os.Stat(o.Flag); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("resolver touched filesystem: %v", err)
	}
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, rel := range []string{".", "agents", "agents/c", "agents/c/n"} {
		info, err := os.Lstat(filepath.Join(o.Flag, rel))
		if err != nil {
			t.Fatal(err)
		}
		if err := checkPrivate(info, 0700, false); err != nil {
			t.Fatal(err)
		}
	}
	for _, rel := range []string{"clusters", "principals", "workers", "staging"} {
		if _, err := os.Stat(filepath.Join(o.Flag, rel)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("created sibling %s", rel)
		}
	}
}

func TestDiscovery(t *testing.T) {
	o := Options{Flag: filepath.Join(privateTemp(t), "home")}
	query := Context{Kind: Principal, ClusterID: "c", Role: "operator"}
	if _, err := Discover(o, query); !errors.Is(err, ErrNotFound) {
		t.Fatal(err)
	}
	if _, err := os.Stat(o.Flag); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("discovery created state")
	}
	first := resolved(t, o, Context{Kind: Principal, ClusterID: "c", Role: "operator", ID: "p1"})
	if err := first.Ensure(); err != nil {
		t.Fatal(err)
	}
	for _, c := range []Context{
		{Kind: Principal, ClusterID: "other", Role: "operator", ID: "p"},
		{Kind: Principal, ClusterID: "c", Role: "other", ID: "p"},
		{Kind: Agent, ClusterID: "c", ID: "p"},
	} {
		if err := resolved(t, o, c).Ensure(); err != nil {
			t.Fatal(err)
		}
	}
	p, err := Discover(o, query)
	if err != nil || p != first {
		t.Fatalf("%#v %v", p, err)
	}
	second := resolved(t, o, Context{Kind: Principal, ClusterID: "c", Role: "operator", ID: "p2"})
	if err := second.Ensure(); err != nil {
		t.Fatal(err)
	}
	if _, err := Discover(o, query); !errors.Is(err, ErrAmbiguous) {
		t.Fatal(err)
	}
	if _, err := Discover(o, Context{Kind: Principal}); !errors.Is(err, ErrAmbiguous) {
		t.Fatal(err)
	}
	p, err = Discover(Options{ExactDir: first.Dir(), Flag: "/ignored"}, query)
	if err != nil || p.Dir() != first.Dir() {
		t.Fatalf("%#v %v", p, err)
	}
	query.ID = "p2"
	p, err = Discover(o, query)
	if err != nil || p != second {
		t.Fatalf("%#v %v", p, err)
	}
}

func TestDiscoveryAllKinds(t *testing.T) {
	for _, c := range []Context{
		{Kind: Authority, ClusterID: "c"}, {Kind: Agent, ClusterID: "c", ID: "n"},
		{Kind: Worker, ClusterID: "c", ID: "w"}, {Kind: Enrollment, ID: "random"},
	} {
		o := Options{Flag: filepath.Join(privateTemp(t), "home")}
		p := resolved(t, o, c)
		if err := p.Ensure(); err != nil {
			t.Fatal(err)
		}
		got, err := Discover(o, Context{Kind: c.Kind})
		if err != nil || got != p {
			t.Fatalf("%#v: %#v %v", c, got, err)
		}
	}
}

func TestUnsafeDirectories(t *testing.T) {
	for _, unsafe := range []string{"symlink", "file", "mode", "ancestor-symlink", "managed-parent-mode"} {
		t.Run(unsafe, func(t *testing.T) {
			base := privateTemp(t)
			o := Options{Flag: filepath.Join(base, "home")}
			c := Context{Kind: Worker, ClusterID: "c", ID: "w"}
			p := resolved(t, o, c)
			if err := p.Ensure(); err != nil {
				t.Fatal(err)
			}
			switch unsafe {
			case "symlink":
				if err := os.Remove(p.Dir()); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(base, p.Dir()); err != nil {
					t.Fatal(err)
				}
			case "file":
				if err := os.Remove(p.Dir()); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p.Dir(), nil, 0600); err != nil {
					t.Fatal(err)
				}
			case "mode":
				if err := os.Chmod(p.Dir(), 0750); err != nil {
					t.Fatal(err)
				}
			case "ancestor-symlink":
				alias := filepath.Join(base, "alias")
				if err := os.Symlink(o.Flag, alias); err != nil {
					t.Fatal(err)
				}
				o.Flag = alias
				p = resolved(t, o, c)
			case "managed-parent-mode":
				if err := os.Chmod(filepath.Dir(p.Dir()), 0755); err != nil {
					t.Fatal(err)
				}
			}
			for _, fn := range []func() error{p.Validate, p.Ensure} {
				if err := fn(); !errors.Is(err, ErrUnsafe) {
					t.Fatalf("expected unsafe: %v", err)
				}
			}
			if _, err := Discover(o, Context{Kind: Worker, ClusterID: "c"}); !errors.Is(err, ErrUnsafe) {
				t.Fatal(err)
			}
			if _, err := Discover(Options{ExactDir: p.Dir()}, Context{Kind: Worker}); !errors.Is(err, ErrUnsafe) && unsafe != "managed-parent-mode" {
				t.Fatal(err)
			}
		})
	}
}

func TestProtectedFiles(t *testing.T) {
	for _, unsafe := range []string{"symlink", "hardlink", "directory", "fifo", "mode", "setuid"} {
		t.Run(unsafe, func(t *testing.T) {
			p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
			target := filepath.Join(p.Dir(), "secret")
			source := filepath.Join(p.Dir(), "source")
			if err := os.WriteFile(source, []byte("unchanged"), 0600); err != nil {
				t.Fatal(err)
			}
			var err error
			switch unsafe {
			case "symlink":
				err = os.Symlink(source, target)
			case "hardlink":
				err = os.Link(source, target)
			case "directory":
				err = os.Mkdir(target, 0700)
			case "fifo":
				err = syscall.Mkfifo(target, 0600)
			case "mode", "setuid":
				err = os.WriteFile(target, []byte("old"), 0600)
				if err == nil {
					mode := os.FileMode(0640)
					if unsafe == "setuid" {
						mode = 0600 | os.ModeSetuid
					}
					err = os.Chmod(target, mode)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			if unsafe == "setuid" {
				info, err := os.Lstat(target)
				if err != nil {
					t.Fatal(err)
				}
				if info.Mode()&os.ModeSetuid == 0 {
					t.Skip("filesystem or sandbox stripped setuid; special-bit validation is tested directly")
				}
			}
			if _, err := p.ReadFile("secret"); !errors.Is(err, ErrUnsafe) {
				t.Fatalf("read: %v", err)
			}
			if err := p.WriteFile("secret", []byte("new")); !errors.Is(err, ErrUnsafe) {
				t.Fatalf("write: %v", err)
			}
			got, err := os.ReadFile(source)
			if err != nil || string(got) != "unchanged" {
				t.Fatalf("changed source: %q %v", got, err)
			}
		})
	}
}

func TestFileNamesAndZeroPath(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Agent})
	for _, name := range []string{"", ".", "..", "../escape", "/absolute", "a/b", "a\\b", temporaryPrefix + "tmp"} {
		if _, err := p.ReadFile(name); !errors.Is(err, ErrUnsafe) {
			t.Fatal(name, err)
		}
		if err := p.WriteFile(name, nil); !errors.Is(err, ErrUnsafe) {
			t.Fatal(name, err)
		}
	}
	if err := (Path{}).Ensure(); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
	if err := p.Publish(nil); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
}

type ownerInfo struct {
	os.FileInfo
	stat syscall.Stat_t
}

type modeInfo struct {
	os.FileInfo
	mode os.FileMode
}

func (i modeInfo) Mode() os.FileMode { return i.mode }

func TestCheckPrivateRejectsSpecialBits(t *testing.T) {
	path := filepath.Join(privateTemp(t), "protected")
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkPrivate(info, 0600, true); err != nil {
		t.Fatalf("baseline: %v", err)
	}
	for _, bit := range []os.FileMode{os.ModeSetuid, os.ModeSetgid, os.ModeSticky} {
		// Inject only the mode bit: ownership and file type still come from a
		// real protected file, independently of filesystem special-bit support.
		withBit := modeInfo{FileInfo: info, mode: info.Mode() | bit}
		if err := checkPrivate(withBit, 0600, true); !errors.Is(err, ErrUnsafe) {
			t.Fatalf("special bit %v accepted: %v", bit, err)
		}
	}
}

func (i ownerInfo) Sys() any { return &i.stat }

func TestOwnershipValidation(t *testing.T) {
	path := privateTemp(t)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	stat := *info.Sys().(*syscall.Stat_t)
	stat.Uid = uint32(os.Geteuid() + 1)
	if err := checkPrivate(ownerInfo{info, stat}, 0700, false); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
	if err := checkAncestor(ownerInfo{info, stat}); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
	// Also exercise the real filesystem when the test user can change ownership.
	if os.Geteuid() == 0 {
		if err := os.Chown(path, 1, -1); err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := os.Chown(path, 0, -1); err != nil {
				t.Error(err)
			}
		}()
		p := resolved(t, Options{ExactDir: path}, Context{Kind: Agent})
		if err := p.Validate(); !errors.Is(err, ErrUnsafe) {
			t.Fatal(err)
		}
	}
}
