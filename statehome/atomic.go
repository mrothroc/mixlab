package statehome

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// WriteFile atomically creates or replaces one protected file. It refuses an
// unsafe existing target, uses a same-directory 0600 temporary, and syncs both
// file and directory. An error after rename can mean publication succeeded but
// its durability could not be confirmed; callers must not assume rollback.
func (p Path) WriteFile(name string, data []byte) (result error) {
	if err := segment(name); err != nil {
		return err
	}
	dir, err := p.lock()
	if err != nil {
		return err
	}
	defer release(dir)
	return p.writeFileLocked(dir, name, data)
}

func (p Path) writeFileLocked(dir *os.File, name string, data []byte) (result error) {
	return p.writeStreamLocked(dir, name, func(w io.Writer) error { _, err := w.Write(data); return err })
}

func (p Path) writeStreamLocked(dir *os.File, name string, write func(io.Writer) error) (result error) {
	target := filepath.Join(p.dir, name)
	if err := validateExistingFile(target); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(p.dir, temporaryPrefix)
	if err != nil {
		return err
	}
	defer func() {
		if err := os.Remove(tmp.Name()); err != nil && !errors.Is(err, os.ErrNotExist) {
			result = errors.Join(result, fmt.Errorf("remove temporary file: %w", err))
		}
	}()
	defer func() { _ = tmp.Close() }()
	info, err := tmp.Stat()
	if err != nil {
		return err
	}
	if err := checkPrivate(info, 0600, true); err != nil {
		return err
	}
	if err := checkFileACL(tmp); err != nil {
		return err
	}
	if err := write(tmp); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := p.sameDirectory(dir); err != nil {
		return err
	}
	if err := validateExistingFile(target); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), target); err != nil {
		return err
	}
	return dir.Sync()
}

func validateExistingFile(path string) error {
	f, err := openProtected(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	return f.Close()
}

// Publish populates an owner-only random sibling staging directory, validates
// and syncs its entire tree, then atomically renames it to this path. Existing
// destinations (including empty directories) are never replaced by cooperating
// publishers. Populate must finish all writes before returning. On failure before
// rename staging cleanup is attempted and cleanup failures are returned; missing
// parents may remain. The final context must
// not already exist. Discovery ignores this package's reserved staging prefix.
// As with WriteFile, a post-rename sync error does not imply rollback.
func (p Path) Publish(populate func(Path) error) (result error) {
	if populate == nil || p.dir == "" {
		return fmt.Errorf("%w: missing path or populate function", ErrUnsafe)
	}
	parent := Path{filepath.Dir(p.dir), p.root, p.kind}
	if err := parent.Ensure(); err != nil {
		return err
	}
	dir, err := parent.lock()
	if err != nil {
		return err
	}
	defer release(dir)
	if err := absent(p.dir); err != nil {
		return err
	}
	name, err := os.MkdirTemp(parent.dir, temporaryPrefix)
	if err != nil {
		return err
	}
	defer func() {
		if err := os.RemoveAll(name); err != nil {
			result = errors.Join(result, fmt.Errorf("remove staging directory: %w", err))
		}
	}()
	stage := Path{name, name, p.kind}
	if err := stage.Validate(); err != nil {
		return err
	}
	if err := populate(stage); err != nil {
		return err
	}
	if err := syncTree(stage); err != nil {
		return err
	}
	if err := parent.sameDirectory(dir); err != nil {
		return err
	}
	if err := absent(p.dir); err != nil {
		return err
	}
	if err := os.Rename(name, p.dir); err != nil {
		return err
	}
	return dir.Sync()
}

func absent(path string) error {
	_, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("%w: %s", ErrExists, path)
}

func (p Path) lock() (*os.File, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	f, err := openNoFollow(p.dir)
	if err != nil {
		return nil, err
	}
	if err := lockDirectory(f); err != nil {
		_ = f.Close()
		return nil, err
	}
	if err := p.sameDirectory(f); err != nil {
		release(f)
		return nil, err
	}
	return f, nil
}

func (p Path) sameDirectory(f *os.File) error {
	if err := p.Validate(); err != nil {
		return err
	}
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	named, err := os.Lstat(p.dir)
	if err != nil {
		return err
	}
	if !os.SameFile(opened, named) {
		return fmt.Errorf("%w: directory changed", ErrUnsafe)
	}
	return checkFileACL(f)
}

func release(f *os.File) { unlockDirectory(f); _ = f.Close() }

func syncTree(p Path) error {
	if err := p.Validate(); err != nil {
		return err
	}
	entries, err := os.ReadDir(p.dir)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if err := segment(entry.Name()); err != nil {
			return err
		}
		name := filepath.Join(p.dir, entry.Name())
		if entry.IsDir() {
			if err := syncTree(Path{name, p.root, p.kind}); err != nil {
				return err
			}
		} else {
			f, err := openProtected(name)
			if err != nil {
				return err
			}
			err = f.Sync()
			closeErr := f.Close()
			if err != nil {
				return err
			}
			if closeErr != nil {
				return closeErr
			}
		}
	}
	return syncDirectory(p.dir)
}

func syncDirectory(path string) error {
	f, err := openNoFollow(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return f.Sync()
}
