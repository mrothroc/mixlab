//go:build darwin || linux

package local

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
)

func TestStoreStagesAndVerifiesImmutableBytes(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		t.Fatal(err)
	}
	s, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte("verified checkpoint bytes")
	h := sha256.Sum256(body)
	ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(body))}
	ctx := context.Background()
	if err := s.Put(ctx, ref, bytes.NewReader([]byte("incomplete"))); err == nil {
		t.Fatal("published incomplete artifact")
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatal("partial artifact leaked", entries, err)
	}
	if err := s.Put(ctx, ref, bytes.NewReader(body)); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := s.Copy(ctx, ref, &out); err != nil || !bytes.Equal(out.Bytes(), body) {
		t.Fatal(out.String(), err)
	}
	if err := s.Put(ctx, ref, bytes.NewReader(body)); !errors.Is(err, statehome.ErrExists) {
		t.Fatal("overwrote artifact", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ref.SHA256), bytes.Repeat([]byte("x"), len(body)), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Copy(ctx, ref, &out); err == nil {
		t.Fatal("trusted digest-named corrupt file")
	}
}
