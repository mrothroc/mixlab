//go:build darwin || linux

package statehome

import (
	"os"
	"path/filepath"
	"testing"
)

func TestTemporaryCleanupPreservesPublishedAndDirectories(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Worker})
	if err := p.WriteFile("published", []byte("keep")); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(p.Dir(), temporaryPrefix+"directory"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(p.Dir(), temporaryPrefix+"orphan"), []byte("partial"), 0600); err != nil {
		t.Fatal(err)
	}
	if n, err := p.CleanupTemporaryFiles(); err != nil || n != 1 {
		t.Fatal(n, err)
	}
	entries, err := os.ReadDir(p.Dir())
	if err != nil || len(entries) != 2 {
		t.Fatal(entries, err)
	}
	if err := os.Symlink(filepath.Join(p.Dir(), "published"), filepath.Join(p.Dir(), temporaryPrefix+"symlink")); err != nil {
		t.Fatal(err)
	}
	if _, err := p.CleanupTemporaryFiles(); err == nil {
		t.Fatal("cleanup followed unsafe symlink")
	}
	if b, err := p.ReadFile("published"); err != nil || string(b) != "keep" {
		t.Fatal("published content lost", err)
	}
}
