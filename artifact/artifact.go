// Package artifact defines immutable byte identities and bounded verified
// transfer. Digests never confer access; owning contexts authorize every use.
package artifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
)

const MaxBytes uint64 = 1 << 30

type Ref struct {
	SHA256 string `json:"sha256"`
	Bytes  uint64 `json:"bytes"`
}

func (r Ref) Validate() error {
	b, err := hex.DecodeString(r.SHA256)
	if err != nil || len(b) != sha256.Size || hex.EncodeToString(b) != r.SHA256 || r.Bytes == 0 || r.Bytes > MaxBytes {
		return fmt.Errorf("artifact requires a canonical SHA256 and size in [1,1GiB]")
	}
	return nil
}

// Copy verifies exactly the declared bytes, then EOF. A caller must stage its
// destination and publish only after this succeeds. It allocates a fixed buffer.
func Copy(ctx context.Context, dst io.Writer, src io.Reader, ref Ref) error {
	if err := ref.Validate(); err != nil {
		return err
	}
	if dst == nil || src == nil {
		return fmt.Errorf("artifact source and destination required")
	}
	hash := sha256.New()
	buffer := make([]byte, 64<<10)
	remaining := ref.Bytes
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := buffer[:min(uint64(len(buffer)), remaining)]
		n, err := io.ReadFull(src, chunk)
		if err != nil {
			return fmt.Errorf("artifact body truncated: %w", err)
		}
		if _, err := hash.Write(chunk[:n]); err != nil {
			return err
		}
		written, err := dst.Write(chunk[:n])
		if err != nil {
			return err
		}
		if written != n {
			return io.ErrShortWrite
		}
		remaining -= uint64(n)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	var extra [1]byte
	if _, err := io.ReadFull(src, extra[:]); err != io.EOF {
		return fmt.Errorf("artifact body exceeds size or missing clean EOF")
	}
	if hex.EncodeToString(hash.Sum(nil)) != ref.SHA256 {
		return fmt.Errorf("artifact checksum mismatch")
	}
	return ctx.Err()
}
