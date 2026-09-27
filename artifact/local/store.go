// Package local stores immutable verified artifacts in an owner-approved
// private directory. It has no remote API or grant policy.
package local

import (
	"context"
	"io"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
)

type Store struct{ path statehome.Path }

func Open(path statehome.Path) (*Store, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Store{path: path}, nil
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
