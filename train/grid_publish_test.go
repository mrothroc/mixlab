//go:build darwin || linux

package train

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGridPublishNeverReplacesDestination(t *testing.T) {
	for _, kind := range []string{"absent", "directory", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			dir := t.TempDir()
			from, to := filepath.Join(dir, "stage"), filepath.Join(dir, "result")
			if err := os.Mkdir(from, 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(from, "new"), []byte("prediction"), 0600); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "directory":
				if err := os.Mkdir(to, 0700); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := os.Symlink("missing", to); err != nil {
					t.Fatal(err)
				}
			}
			err := publishGridDirectory(from, to)
			if kind == "absent" {
				if err != nil {
					t.Fatal(err)
				}
				if _, err := os.Stat(filepath.Join(to, "new")); err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatal("replaced existing destination")
			}
			if _, err := os.Stat(filepath.Join(from, "new")); err != nil {
				t.Fatal("lost staged prediction", err)
			}
			if kind == "symlink" {
				if target, err := os.Readlink(to); err != nil || target != "missing" {
					t.Fatal("changed symlink", target, err)
				}
			} else if entries, err := os.ReadDir(to); err != nil || len(entries) != 0 {
				t.Fatal("changed destination", entries, err)
			}
		})
	}
}
