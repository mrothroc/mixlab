package clusterapp

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/statehome"
)

func TestCheckpointInputChunks(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Agent})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Ensure(); err != nil {
		t.Fatal(err)
	}
	r := func(n uint64) ([]byte, error) { return io.ReadAll(&inputChunks{path: p, size: n}) }
	if _, err := r(3); err == nil {
		t.Fatal("missing chunk accepted")
	}
	if err := immutableInputBytes(p, "chunk-000000", []byte("abc")); err != nil {
		t.Fatal(err)
	}
	if err := immutableInputBytes(p, "chunk-000000", []byte("abc")); err != nil {
		t.Fatal("exact retry", err)
	}
	if err := immutableInputBytes(p, "chunk-000000", []byte("xyz")); err == nil {
		t.Fatal("changed retry accepted")
	}
	for _, size := range []uint64{2, 4, outputChunkBytes + 1} {
		if _, err := r(size); err == nil {
			t.Fatalf("accepted incorrect chunk size for %d bytes", size)
		}
	}
	b, err := r(3)
	if err != nil || !bytes.Equal(b, []byte("abc")) {
		t.Fatal(string(b), err)
	}
}
