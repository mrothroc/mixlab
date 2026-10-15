//go:build darwin || linux

package local

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
)

type crashReader struct{ reads int }

func (r *crashReader) Read(b []byte) (int, error) {
	r.reads++
	if r.reads > 1 {
		os.Exit(77)
	}
	clear(b)
	return len(b), nil
}

func TestArtifactCrashStagingCleanup(t *testing.T) {
	dir := os.Getenv("MIXLAB_P3_ARTIFACT_CRASH_DIR")
	child := dir != ""
	if !child {
		dir = t.TempDir()
	}
	dir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	body := make([]byte, 128<<10)
	h := sha256.Sum256(body)
	ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(body))}
	if child {
		s, err := Open(p)
		if err != nil {
			t.Fatal(err)
		}
		_ = s.Put(context.Background(), ref, &crashReader{})
		t.Fatal("did not crash")
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestArtifactCrashStagingCleanup$")
	cmd.Env = append(os.Environ(), "MIXLAB_P3_ARTIFACT_CRASH_DIR="+dir)
	if err = cmd.Run(); err == nil {
		t.Fatal("child did not terminate")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 1 {
		t.Fatal("expected one interrupted staging file", entries, err)
	}
	s, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	entries, err = os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatal("staging survived recovery", entries, err)
	}
	if err = s.PutVerified(context.Background(), ref, bytes.NewReader(body)); err != nil {
		t.Fatal("retry failed", err)
	}
}
