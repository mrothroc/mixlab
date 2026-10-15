package local

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

type Artifacts struct{ Store *artifactlocal.Store }

func (a Artifacts) Load(ctx context.Context, b run.Blob) ([]optimizer.Tensor, error) {
	if a.Store == nil {
		return nil, fmt.Errorf("tensor artifact store required")
	}
	if err := tensorKind(b.Kind); err != nil {
		return nil, err
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	f, err := a.Store.OpenVerified(ctx, b.Ref)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return tensors.Read(f, b.Ref.Bytes)
}
func tensorKind(k string) error {
	if k != "model_f32" && k != "delta_f32" && k != "outer_state_f32" {
		return fmt.Errorf("unsupported global tensor artifact")
	}
	return nil
}
func (a Artifacts) Put(ctx context.Context, kind string, ts []optimizer.Tensor) (_ run.Blob, result error) {
	if a.Store == nil {
		return run.Blob{}, fmt.Errorf("tensor artifact store required")
	}
	if err := tensorKind(kind); err != nil {
		return run.Blob{}, err
	}
	if err := ctx.Err(); err != nil {
		return run.Blob{}, err
	}
	// Spool once to a private bounded temporary, avoiding another model-sized RAM
	// copy. Only the verified content-addressed destination survives this call.
	f, err := os.CreateTemp("", "mixlab-coordinator-tensors-*")
	if err != nil {
		return run.Blob{}, err
	}
	// Unlink the open spool immediately on supported Unix hosts: a process crash
	// must not strand a model-sized temporary in the shared temporary directory.
	if err = os.Remove(f.Name()); err != nil {
		return run.Blob{}, errors.Join(err, f.Close())
	}
	defer func() { result = errors.Join(result, f.Close()) }()
	h := sha256.New()
	if err = tensors.Write(io.MultiWriter(f, h), ts); err != nil {
		return run.Blob{}, err
	}
	size, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return run.Blob{}, err
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return run.Blob{}, err
	}
	b := run.Blob{Kind: kind, Ref: artifact.Ref{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: uint64(size)}}
	if err = b.Validate(); err != nil {
		return run.Blob{}, err
	}
	return b, a.Store.PutVerified(ctx, b.Ref, f)
}
