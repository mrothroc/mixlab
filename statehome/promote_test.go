package statehome

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPromoteDurableStage(t *testing.T) {
	root := privateTemp(t)
	stage := resolved(t, Options{ExactDir: filepath.Join(root, "stage")}, Context{Kind: Enrollment})
	dest := resolved(t, Options{ExactDir: filepath.Join(root, "final")}, Context{Kind: Principal})
	if err := stage.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := stage.WriteFile("record", []byte("synthetic")); err != nil {
		t.Fatal(err)
	}
	if err := dest.Promote(stage); err != nil {
		t.Fatal(err)
	}
	got, err := dest.ReadFile("record")
	if err != nil || string(got) != "synthetic" {
		t.Fatal(string(got), err)
	}
	if _, err := os.Lstat(stage.Dir()); !os.IsNotExist(err) {
		t.Fatal("stage remains", err)
	}
	if err := stage.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := stage.WriteFile("record", []byte("replacement")); err != nil {
		t.Fatal(err)
	}
	if err := dest.Promote(stage); err == nil {
		t.Fatal("replaced existing context")
	}
	got, err = dest.ReadFile("record")
	if err != nil || string(got) != "synthetic" {
		t.Fatal("destination changed")
	}
	if err := stage.Promote(stage); err == nil {
		t.Fatal("self promotion")
	}
}

func TestPromoteRejectsUnsafeStage(t *testing.T) {
	root := privateTemp(t)
	stage := resolved(t, Options{ExactDir: filepath.Join(root, "stage")}, Context{Kind: Enrollment})
	dest := resolved(t, Options{ExactDir: filepath.Join(root, "final")}, Context{Kind: Principal})
	if err := stage.Ensure(); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/etc/passwd", filepath.Join(stage.Dir(), "link")); err != nil {
		t.Fatal(err)
	}
	if err := dest.Promote(stage); err == nil {
		t.Fatal("promoted unsafe tree")
	}
	if _, err := os.Lstat(dest.Dir()); !os.IsNotExist(err) {
		t.Fatal("unsafe destination published")
	}
}
