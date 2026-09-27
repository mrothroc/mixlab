package statehome

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// chmod is fixture setup only. Production reads ACLs through native syscalls.
func addDarwinACL(t *testing.T, path, entry string) {
	t.Helper()
	if out, err := exec.Command("/bin/chmod", "+a", entry, path).CombinedOutput(); err != nil {
		t.Fatalf("ACL fixture: %v: %s", err, out)
	}
	t.Cleanup(func() {
		if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
			return
		}
		if out, err := exec.Command("/bin/chmod", "-N", path).CombinedOutput(); err != nil {
			t.Errorf("ACL fixture cleanup: %v: %s", err, out)
		}
	})
}

func requireMode(t *testing.T, path string, mode os.FileMode) {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != mode {
		t.Fatalf("mode %o, want %o", info.Mode().Perm(), mode)
	}
}

func TestDarwinACLProtectedFile(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	if err := p.WriteFile("key", []byte("original")); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(p.Dir(), "key")
	addDarwinACL(t, path, "everyone allow read")
	requireMode(t, path, 0600)
	if _, err := p.ReadFile("key"); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("read: %v", err)
	}
	if err := p.WriteFile("key", []byte("replacement")); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("write: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "original" {
		t.Fatalf("modified target: %q %v", data, err)
	}
}

func TestDarwinACLDirectoriesAndAncestors(t *testing.T) {
	for _, ancestor := range []bool{false, true} {
		t.Run(map[bool]string{false: "context", true: "ancestor"}[ancestor], func(t *testing.T) {
			base := privateTemp(t)
			p := resolved(t, Options{ExactDir: filepath.Join(base, "state")}, Context{Kind: Agent})
			if err := p.Ensure(); err != nil {
				t.Fatal(err)
			}
			target := p.Dir()
			if ancestor {
				target = base
			}
			addDarwinACL(t, target, "everyone allow list,search")
			requireMode(t, target, 0700)
			for _, op := range []func() error{p.Validate, p.Ensure, func() error { return p.WriteFile("key", nil) }} {
				if err := op(); !errors.Is(err, ErrUnsafe) {
					t.Fatal(err)
				}
			}
			if _, err := Discover(Options{ExactDir: p.Dir()}, Context{Kind: Agent}); !errors.Is(err, ErrUnsafe) {
				t.Fatal(err)
			}
			child := resolved(t, Options{ExactDir: filepath.Join(p.Dir(), "child")}, Context{Kind: Worker})
			if err := child.Publish(func(Path) error { t.Error("populated below unsafe ACL"); return nil }); !errors.Is(err, ErrUnsafe) {
				t.Fatal(err)
			}
		})
	}
}

func TestDarwinACLInheritedChildren(t *testing.T) {
	base := privateTemp(t)
	addDarwinACL(t, base, "everyone allow read,file_inherit,directory_inherit,only_inherit")
	if err := checkPathACL(base); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("inherit-only grant accepted: %v", err)
	}
	dir, err := os.MkdirTemp(base, temporaryPrefix)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.CreateTemp(base, temporaryPrefix)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	// Remove the parent's ACL to prove the children are rejected on their own.
	if out, err := exec.Command("/bin/chmod", "-N", base).CombinedOutput(); err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	requireMode(t, dir, 0700)
	requireMode(t, f.Name(), 0600)
	if err := resolved(t, Options{ExactDir: dir}, Context{Kind: Enrollment}).Validate(); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("staging ACL: %v", err)
	}
	if err := checkFileACL(f); !errors.Is(err, ErrUnsafe) {
		t.Fatalf("temporary file ACL: %v", err)
	}
}

func TestDarwinACLRejectsStagedGrants(t *testing.T) {
	for _, directory := range []bool{false, true} {
		base := privateTemp(t)
		p := resolved(t, Options{ExactDir: filepath.Join(base, "final")}, Context{Kind: Worker})
		err := p.Publish(func(stage Path) error {
			if err := stage.WriteFile("key", []byte("private")); err != nil {
				return err
			}
			target := filepath.Join(stage.Dir(), "key")
			if directory {
				target = stage.Dir()
			}
			addDarwinACL(t, target, "everyone allow read")
			return nil
		})
		if !errors.Is(err, ErrUnsafe) {
			t.Fatal(err)
		}
		entries, err := os.ReadDir(base)
		if err != nil || len(entries) != 0 {
			t.Fatalf("unsafe staging published or leaked: %v %v", entries, err)
		}
	}
}

func TestDarwinACLDenyOnlyHomeDefault(t *testing.T) {
	home := privateTemp(t)
	addDarwinACL(t, home, "everyone deny delete")
	p := resolved(t, Options{UserHome: home}, Context{Kind: Principal, ClusterID: "c", Role: "r", ID: "p"})
	if err := p.Publish(func(stage Path) error { return stage.WriteFile("key", []byte("private")) }); err != nil {
		t.Fatal(err)
	}
	if _, err := p.ReadFile("key"); err != nil {
		t.Fatal(err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	// A deny-only ACL on protected data is harmless as well.
	addDarwinACL(t, filepath.Join(p.Dir(), "key"), "everyone deny delete")
	if _, err := p.ReadFile("key"); err != nil {
		t.Fatal(err)
	}
}

func TestDarwinACLInspectionErrors(t *testing.T) {
	base := privateTemp(t)
	if err := checkPathACL(filepath.Join(base, "missing")); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
	f, err := os.CreateTemp(base, temporaryPrefix)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := checkFileACL(f); !errors.Is(err, ErrUnsafe) {
		t.Fatal(err)
	}
}
