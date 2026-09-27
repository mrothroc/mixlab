//go:build darwin || linux

package statehome

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestPublishStreamAtomicBoundedAndImmutable(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Worker})
	for _, mode := range []string{"callback-error", "ignored-overflow", "success"} {
		err := p.PublishStream("blob", 4, func(w io.Writer) error {
			if _, err := os.Stat(filepath.Join(p.Dir(), "blob")); !os.IsNotExist(err) {
				t.Fatal("published before validation", err)
			}
			if mode == "ignored-overflow" {
				_, _ = w.Write([]byte("oversized"))
				return nil
			}
			_, err := w.Write([]byte("data"))
			if mode == "callback-error" {
				return errors.New("checksum failed")
			}
			return err
		})
		if (err == nil) != (mode == "success") {
			t.Fatal(mode, err)
		}
		entries, err := os.ReadDir(p.Dir())
		if err != nil {
			t.Fatal(err)
		}
		want := 0
		if mode == "success" {
			want = 1
		}
		if len(entries) != want {
			t.Fatal("leaked temporary", entries)
		}
	}
	if err := p.PublishStream("blob", 4, func(io.Writer) error { t.Fatal("existing target overwritten"); return nil }); !errors.Is(err, ErrExists) {
		t.Fatal(err)
	}
	r, err := p.OpenRead("blob")
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(r)
	if closeErr := r.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil || !bytes.Equal(b, []byte("data")) {
		t.Fatal(string(b), err)
	}
}

func TestStreamRejectsUnsafeNamesAndLinks(t *testing.T) {
	p := resolved(t, Options{ExactDir: privateTemp(t)}, Context{Kind: Worker})
	for _, name := range []string{"../escape", "/absolute", ""} {
		if err := p.PublishStream(name, 4, func(io.Writer) error { t.Fatal("unsafe callback invoked"); return nil }); err == nil {
			t.Fatal(name)
		}
		if r, err := p.OpenRead(name); err == nil {
			_ = r.Close()
			t.Fatal(name)
		}
	}
	if err := os.Symlink("missing", filepath.Join(p.Dir(), "link")); err != nil {
		t.Fatal(err)
	}
	if err := p.PublishStream("link", 4, func(io.Writer) error { t.Fatal("link replaced"); return nil }); err == nil {
		t.Fatal("link accepted")
	}
	if r, err := p.OpenRead("link"); err == nil {
		_ = r.Close()
		t.Fatal("link read")
	}
}
