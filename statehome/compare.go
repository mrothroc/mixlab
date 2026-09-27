package statehome

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

var ErrConflict = errors.New("protected state changed; reload before retry")

// ReadFileLimit bounds allocation even if a protected file is corrupt or grows
// during the read. Existing ownership, inode, permission, and ACL checks apply.
func (p Path) ReadFileLimit(name string, limit int64) ([]byte, error) {
	if limit <= 0 || limit == int64(^uint64(0)>>1) {
		return nil, fmt.Errorf("invalid protected-file size limit")
	}
	if err := segment(name); err != nil {
		return nil, err
	}
	dir, err := p.lock()
	if err != nil {
		return nil, err
	}
	defer release(dir)
	f, err := openProtected(filepath.Join(p.dir, name))
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("protected file exceeds size limit")
	}
	return b, nil
}

// CompareAndSwap atomically changes one file only if its bytes equal expected.
// A nil expected requires absence; a nil replacement deletes an existing file.
// A non-nil empty slice denotes an existing empty file. Cooperating processes
// use the directory lock, so a stale writer cannot overwrite newer state.
// As with WriteFile, a post-publication sync error does not imply rollback.
func (p Path) CompareAndSwap(name string, expected, replacement []byte) error {
	if err := segment(name); err != nil {
		return err
	}
	if expected == nil && replacement == nil {
		return fmt.Errorf("empty state transition")
	}
	dir, err := p.lock()
	if err != nil {
		return err
	}
	defer release(dir)
	f, err := openProtected(filepath.Join(p.dir, name))
	switch {
	case errors.Is(err, os.ErrNotExist):
		if expected != nil {
			return ErrConflict
		}
	case err != nil:
		return err
	default:
		if expected == nil {
			_ = f.Close()
			return ErrConflict
		}
		got, readErr := io.ReadAll(io.LimitReader(f, int64(len(expected))+1))
		closeErr := f.Close()
		if err := errors.Join(readErr, closeErr); err != nil {
			return err
		}
		if !bytes.Equal(got, expected) {
			return ErrConflict
		}
	}
	if replacement != nil {
		return p.writeFileLocked(dir, name, replacement)
	}
	if err := p.sameDirectory(dir); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(p.dir, name)); err != nil {
		return err
	}
	return dir.Sync()
}
