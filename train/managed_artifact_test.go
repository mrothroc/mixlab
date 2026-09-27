package train

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"io"
	"testing"
)

func TestManagedArtifactHashHonorsCancellation(t *testing.T) {
	payload := bytes.Repeat([]byte("weights"), 30000)
	h := sha256.New()
	n, err := hashManagedArtifact(context.Background(), h, bytes.NewReader(payload))
	want := sha256.Sum256(payload)
	if err != nil || n != int64(len(payload)) || !bytes.Equal(h.Sum(nil), want[:]) {
		t.Fatal(n, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &cancelHashReader{Reader: bytes.NewReader(payload), cancel: cancel}
	n, err = hashManagedArtifact(ctx, io.Discard, r)
	if !errors.Is(err, context.Canceled) || n > 64<<10 || r.reads != 1 {
		t.Fatal(n, err, r.reads)
	}
}

type cancelHashReader struct {
	io.Reader
	cancel context.CancelFunc
	reads  int
}

func (r *cancelHashReader) Read(b []byte) (int, error) {
	r.reads++
	r.cancel()
	return r.Reader.Read(b)
}
