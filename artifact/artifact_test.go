package artifact

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"testing"
)

func TestCopyChecksLengthChecksumAndCancellation(t *testing.T) {
	body := bytes.Repeat([]byte("artifact"), 20000)
	h := sha256.Sum256(body)
	ref := Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(body))}
	for _, test := range []struct {
		name  string
		body  []byte
		valid bool
	}{
		{"exact", body, true}, {"truncated", body[:len(body)-1], false}, {"oversized", append(append([]byte{}, body...), 0), false}, {"corrupt", bytes.Repeat([]byte("x"), len(body)), false},
	} {
		t.Run(test.name, func(t *testing.T) {
			var out bytes.Buffer
			err := Copy(context.Background(), &out, bytes.NewReader(test.body), ref)
			if (err == nil) != test.valid {
				t.Fatal(err)
			}
			if test.valid && !bytes.Equal(out.Bytes(), body) {
				t.Fatal("changed bytes")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := Copy(ctx, io.Discard, bytes.NewReader(body), ref); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := Copy(context.Background(), shortWriter{}, bytes.NewReader(body), ref); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
}

type shortWriter struct{}

func (shortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
