package train

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/workerjob"
)

func sendManagedArtifactFile(ctx context.Context, name string, limit uint64, send func(workerjob.ArtifactMessage) error) error {
	info, err := os.Lstat(name)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() <= 0 || uint64(info.Size()) > limit {
		return fmt.Errorf("final weights exceed approved output budget or are not a regular file")
	}
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	if !os.SameFile(info, opened) {
		return fmt.Errorf("final weights changed during open")
	}
	if err := f.Chmod(0600); err != nil {
		return err
	}
	h := sha256.New()
	n, err := hashManagedArtifact(ctx, h, io.LimitReader(f, int64(limit)+1))
	if err != nil {
		return err
	}
	if n != info.Size() {
		return fmt.Errorf("final weights changed size")
	}
	ref := artifact.Ref{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: uint64(n)}
	if err := ref.Validate(); err != nil {
		return err
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := send(workerjob.ArtifactMessage{Phase: "begin", Ref: ref}); err != nil {
		return err
	}
	buffer := make([]byte, workerjob.ArtifactChunkBytes)
	var offset uint64
	for offset < ref.Bytes {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := buffer[:min(uint64(len(buffer)), ref.Bytes-offset)]
		if _, err := io.ReadFull(f, chunk); err != nil {
			return err
		}
		if err := send(workerjob.ArtifactMessage{Phase: "chunk", Ref: ref, Offset: offset, Data: chunk}); err != nil {
			return err
		}
		offset += uint64(len(chunk))
	}
	return send(workerjob.ArtifactMessage{Phase: "end", Ref: ref, Offset: offset})
}

func hashManagedArtifact(ctx context.Context, hash io.Writer, src io.Reader) (int64, error) {
	buffer := make([]byte, workerjob.ArtifactChunkBytes)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, err := src.Read(buffer)
		if n > 0 {
			written, writeErr := hash.Write(buffer[:n])
			total += int64(written)
			if writeErr != nil {
				return total, writeErr
			}
			if written != n {
				return total, io.ErrShortWrite
			}
		}
		if err == io.EOF {
			return total, ctx.Err()
		}
		if err != nil {
			return total, err
		}
		if n == 0 {
			return total, io.ErrNoProgress
		}
	}
}
