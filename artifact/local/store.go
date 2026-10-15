// Package local stores immutable verified artifacts in an owner-approved
// private directory. It has no remote API or grant policy.
package local

import (
	"context"
	"errors"
	"io"
	"os"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
)

type Store struct{ path statehome.Path }

func Open(path statehome.Path) (*Store, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	if _, err := path.CleanupTemporaryFiles(); err != nil {
		return nil, err
	}
	return &Store{path: path}, nil
}

// PutVerified permits exact retries, but verifies both the supplied stream and
// the existing inode. A digest-shaped filename never proves integrity.
func (s *Store) PutVerified(ctx context.Context, ref artifact.Ref, source io.Reader) error {
	err := s.Put(ctx, ref, source)
	if !errors.Is(err, statehome.ErrExists) {
		return err
	}
	if err := artifact.Copy(ctx, io.Discard, source, ref); err != nil {
		return err
	}
	return s.Copy(ctx, ref, io.Discard)
}

// OpenVerified returns the verified immutable inode positioned at its start.
// The owner must close it. Private state directories must not be modified by
// other applications while their artifacts are in use.
func (s *Store) OpenVerified(ctx context.Context, ref artifact.Ref) (*os.File, error) {
	if err := ref.Validate(); err != nil {
		return nil, err
	}
	f, err := s.path.OpenRead(ref.SHA256)
	if err != nil {
		return nil, err
	}
	if err = artifact.Copy(ctx, io.Discard, f, ref); err == nil {
		_, err = f.Seek(0, io.SeekStart)
	}
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return f, nil
}

// Put never overwrites even an existing same-digest file. Owners may verify an
// existing reference separately on exact retry; a corrupt file is not repaired.
func (s *Store) Put(ctx context.Context, ref artifact.Ref, source io.Reader) error {
	if err := ref.Validate(); err != nil {
		return err
	}
	return s.path.PublishStream(ref.SHA256, int64(ref.Bytes), func(w io.Writer) error {
		return artifact.Copy(ctx, w, source, ref)
	})
}

// Copy verifies stored bytes as it streams them. A remote receiver must stage
// and checksum too; a successful open is not proof of the complete contents.
func (s *Store) Copy(ctx context.Context, ref artifact.Ref, destination io.Writer) error {
	if err := ref.Validate(); err != nil {
		return err
	}
	r, err := s.path.OpenRead(ref.SHA256)
	if err != nil {
		return err
	}
	defer func() { _ = r.Close() }()
	return artifact.Copy(ctx, destination, r, ref)
}
