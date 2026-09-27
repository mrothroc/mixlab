//go:build darwin || linux

package statehome

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestWriteFile(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Principal})
	for _, value := range []string{"first", "replacement", ""} {
		if err := p.WriteFile("key", []byte(value)); err != nil {
			t.Fatal(err)
		}
		got, err := p.ReadFile("key")
		if err != nil || string(got) != value {
			t.Fatalf("%q %v", got, err)
		}
		info, err := os.Lstat(filepath.Join(p.Dir(), "key"))
		if err != nil {
			t.Fatal(err)
		}
		if err := checkPrivate(info, 0600, true); err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(p.Dir())
	if err != nil || len(entries) != 1 {
		t.Fatalf("temporary leak: %v %v", entries, err)
	}
}

func TestPublish(t *testing.T) {
	o := Options{Flag: filepath.Join(privateTemp(t), "home")}
	c := Context{Kind: Principal, ClusterID: "c", Role: "opaque", ID: "p"}
	p := resolved(t, o, c)
	var staging string
	err := p.Publish(func(stage Path) error {
		staging = stage.Dir()
		if stage.Kind() != Principal || filepath.Dir(stage.Dir()) != filepath.Dir(p.Dir()) {
			return fmt.Errorf("incorrect stage: %#v", stage)
		}
		if _, err := os.Lstat(p.Dir()); !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("destination visible before publication: %v", err)
		}
		if _, err := Discover(o, Context{Kind: Principal, ClusterID: "c", Role: "opaque"}); !errors.Is(err, ErrNotFound) {
			return fmt.Errorf("staging discovered: %v", err)
		}
		return stage.WriteFile("key", []byte("private"))
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := p.ReadFile("key")
	if err != nil || string(got) != "private" {
		t.Fatalf("%q %v", got, err)
	}
	if _, err := os.Lstat(staging); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("staging survived: %v", err)
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := Discover(o, c); err != nil {
		t.Fatal(err)
	}
	called := false
	err = p.Publish(func(Path) error { called = true; return nil })
	if !errors.Is(err, ErrExists) || called {
		t.Fatalf("existing identity overwritten: %v", err)
	}
}

func TestPublishNeverReplacesExisting(t *testing.T) {
	for _, kind := range []string{"empty-directory", "file", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			base := privateTemp(t)
			p := resolved(t, Options{ExactDir: filepath.Join(base, "final")}, Context{Kind: Worker})
			var err error
			switch kind {
			case "empty-directory":
				err = os.Mkdir(p.Dir(), 0700)
			case "file":
				err = os.WriteFile(p.Dir(), []byte("existing"), 0600)
			case "symlink":
				err = os.Symlink(filepath.Join(base, "missing"), p.Dir())
			}
			if err != nil {
				t.Fatal(err)
			}
			before, err := os.Lstat(p.Dir())
			if err != nil {
				t.Fatal(err)
			}
			if err := p.Publish(func(Path) error { t.Fatal("populate called"); return nil }); !errors.Is(err, ErrExists) {
				t.Fatal(err)
			}
			after, err := os.Lstat(p.Dir())
			if err != nil || !os.SameFile(before, after) {
				t.Fatal("existing destination replaced")
			}
		})
	}
}

func TestPublishFailureCleansStaging(t *testing.T) {
	failure := errors.New("issuance rejected")
	for _, kind := range []string{"callback", "symlink", "hardlink", "file-mode", "directory-mode"} {
		t.Run(kind, func(t *testing.T) {
			base := privateTemp(t)
			p := resolved(t, Options{ExactDir: filepath.Join(base, "final")}, Context{Kind: Worker})
			err := p.Publish(func(stage Path) error {
				if err := stage.WriteFile("key", []byte("private")); err != nil {
					return err
				}
				switch kind {
				case "callback":
					return failure
				case "symlink":
					return os.Symlink("key", filepath.Join(stage.Dir(), "link"))
				case "hardlink":
					return os.Link(filepath.Join(stage.Dir(), "key"), filepath.Join(stage.Dir(), "link"))
				case "file-mode":
					return os.Chmod(filepath.Join(stage.Dir(), "key"), 0644)
				case "directory-mode":
					return os.Mkdir(filepath.Join(stage.Dir(), "subdir"), 0755)
				}
				return nil
			})
			want := ErrUnsafe
			if kind == "callback" {
				want = failure
			}
			if !errors.Is(err, want) {
				t.Fatalf("got %v want %v", err, want)
			}
			if _, err := os.Lstat(p.Dir()); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("published rejected state: %v", err)
			}
			entries, err := os.ReadDir(base)
			if err != nil || len(entries) != 0 {
				t.Fatalf("staging leaked: %v %v", entries, err)
			}
		})
	}
}

func TestPublishNestedState(t *testing.T) {
	p := resolved(t, Options{ExactDir: filepath.Join(privateTemp(t), "final")}, Context{Kind: Worker})
	if err := p.Publish(func(stage Path) error {
		child := filepath.Join(stage.Dir(), "child")
		if err := os.Mkdir(child, 0700); err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(child, "state"), []byte("nested"), 0600)
	}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(p.Dir(), "child", "state"))
	if err != nil || string(data) != "nested" {
		t.Fatalf("%q %v", data, err)
	}
}

func TestDestinationAppearsDuringStaging(t *testing.T) {
	base := privateTemp(t)
	p := resolved(t, Options{ExactDir: filepath.Join(base, "final")}, Context{Kind: Agent})
	err := p.Publish(func(stage Path) error {
		if err := stage.WriteFile("key", []byte("unpublished")); err != nil {
			return err
		}
		return os.Mkdir(p.Dir(), 0700)
	})
	if !errors.Is(err, ErrExists) {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(p.Dir())
	if err != nil || len(entries) != 0 {
		t.Fatalf("destination replaced: %v %v", entries, err)
	}
	entries, err = os.ReadDir(base)
	if err != nil || len(entries) != 1 {
		t.Fatalf("staging leaked: %v %v", entries, err)
	}
}

func TestConcurrentPublish(t *testing.T) {
	p := resolved(t, Options{ExactDir: filepath.Join(privateTemp(t), "final")}, Context{Kind: Agent})
	const count = 12
	results := make(chan error, count)
	var wg sync.WaitGroup
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results <- p.Publish(func(stage Path) error { return stage.WriteFile("winner", []byte(fmt.Sprint(i))) })
		}(i)
	}
	wg.Wait()
	close(results)
	successes := 0
	for err := range results {
		if err == nil {
			successes++
		} else if !errors.Is(err, ErrExists) {
			t.Fatal(err)
		}
	}
	if successes != 1 {
		t.Fatalf("%d successful publications", successes)
	}
	if _, err := p.ReadFile("winner"); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Dir(p.Dir()))
	if err != nil || len(entries) != 1 {
		t.Fatalf("staging leaked: %v %v", entries, err)
	}
}

func TestAtomicReplacementReaders(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Agent})
	a, b := bytes.Repeat([]byte("a"), 64*1024), bytes.Repeat([]byte("b"), 64*1024)
	if err := p.WriteFile("state", a); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	results := make(chan error, 4)
	for worker := 0; worker < 4; worker++ {
		go func(worker int) {
			<-start
			for i := 0; i < 20; i++ {
				if worker < 2 {
					value := a
					if worker == 1 {
						value = b
					}
					if err := p.WriteFile("state", value); err != nil {
						results <- err
						return
					}
				} else {
					// Read the published pathname directly to verify the OS-level
					// atomicity, independently of ReadFile's inode-race rejection.
					value, err := os.ReadFile(filepath.Join(p.Dir(), "state"))
					if err != nil {
						results <- err
						return
					}
					if !bytes.Equal(value, a) && !bytes.Equal(value, b) {
						results <- fmt.Errorf("partial replacement")
						return
					}
				}
			}
			results <- nil
		}(worker)
	}
	close(start)
	for i := 0; i < 4; i++ {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
	entries, err := os.ReadDir(p.Dir())
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), temporaryPrefix) {
			t.Fatal("temporary leak")
		}
	}
}
