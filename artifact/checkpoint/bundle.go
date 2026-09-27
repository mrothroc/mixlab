// Package checkpoint frames opaque distributed checkpoint members. Numerical
// validation remains in the trainer; no archive member can select a path.
package checkpoint

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/mrothroc/mixlab/artifact"
)

const Manifest = "checkpoint.distributed.resume.json"
const Model = "model.safetensors"
const State = "state.safetensors"
const File = "checkpoint.mixlab"
const ManifestMaxBytes = 4 << 20

var names = [3]string{Manifest, Model, State}
var magic = [8]byte{'M', 'I', 'X', 'D', 'D', 'P', '0', '1'}

type Member struct {
	Size   uint64
	Reader io.Reader
}

func sizesValid(sizes [3]uint64) error {
	total := uint64(32)
	for i, n := range sizes {
		if n == 0 || n > artifact.MaxBytes-total || (i == 0 && n > ManifestMaxBytes) {
			return fmt.Errorf("invalid checkpoint member size")
		}
		total += n
	}
	return nil
}

func Write(ctx context.Context, dst io.Writer, members [3]Member) error {
	var sizes [3]uint64
	for i, m := range members {
		sizes[i] = m.Size
	}
	if err := sizesValid(sizes); err != nil {
		return err
	}
	var header [32]byte
	copy(header[:8], magic[:])
	for i, n := range sizes {
		binary.LittleEndian.PutUint64(header[8+i*8:], n)
	}
	if n, err := dst.Write(header[:]); err != nil {
		return err
	} else if n != len(header) {
		return io.ErrShortWrite
	}
	for _, m := range members {
		if m.Reader == nil {
			return fmt.Errorf("checkpoint reader required")
		}
		if err := copyExact(ctx, dst, m.Reader, m.Size); err != nil {
			return err
		}
		var extra [1]byte
		if n, err := m.Reader.Read(extra[:]); n != 0 || err != io.EOF {
			return fmt.Errorf("checkpoint member has trailing bytes")
		}
	}
	return ctx.Err()
}

// Read calls accept exactly once per fixed member. The callback must consume
// its entire bounded reader. Callers publish only after Read succeeds.
func Read(ctx context.Context, src io.Reader, accept func(string, uint64, io.Reader) error) error {
	if src == nil || accept == nil {
		return fmt.Errorf("checkpoint source and consumer required")
	}
	src = contextReader{ctx, src}
	var header [32]byte
	if _, err := io.ReadFull(src, header[:]); err != nil {
		return err
	}
	if string(header[:8]) != string(magic[:]) {
		return fmt.Errorf("invalid checkpoint container version")
	}
	var sizes [3]uint64
	for i := range sizes {
		sizes[i] = binary.LittleEndian.Uint64(header[8+i*8:])
	}
	if err := sizesValid(sizes); err != nil {
		return err
	}
	for i, n := range sizes {
		if err := ctx.Err(); err != nil {
			return err
		}
		limited := &io.LimitedReader{R: src, N: int64(n)}
		if err := accept(names[i], n, limited); err != nil {
			return err
		}
		if limited.N != 0 {
			return io.ErrUnexpectedEOF
		}
	}
	var extra [1]byte
	if n, err := src.Read(extra[:]); n != 0 || err != io.EOF {
		return fmt.Errorf("checkpoint container has trailing bytes")
	}
	return ctx.Err()
}

type contextReader struct {
	ctx context.Context
	src io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.src.Read(p)
}

func copyExact(ctx context.Context, dst io.Writer, src io.Reader, remaining uint64) error {
	buffer := make([]byte, 64<<10)
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		b := buffer[:min(uint64(len(buffer)), remaining)]
		if _, err := io.ReadFull(src, b); err != nil {
			return err
		}
		n, err := dst.Write(b)
		if err != nil {
			return err
		}
		if n != len(b) {
			return io.ErrShortWrite
		}
		remaining -= uint64(n)
	}
	return nil
}
