package statehome

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Ensure creates only this context's ancestors and directory, never sibling
// contexts. Existing unsafe paths are rejected, not chmod'ed or repaired.
func (p Path) Ensure() error { return p.walk(true) }

// Validate checks the directory and all managed ancestors, without creating
// anything. It does not parse or validate the contents of the directory.
func (p Path) Validate() error { return p.walk(false) }

func (p Path) walk(create bool) error {
	if p.dir == "" || p.root == "" {
		return fmt.Errorf("%w: zero path", ErrUnsafe)
	}
	current := string(filepath.Separator)
	info, err := os.Lstat(current)
	if err != nil {
		return err
	}
	if err := checkDirectory(info, p.root == current); err != nil {
		return err
	}
	if err := checkPathACL(current); err != nil {
		return err
	}
	for _, part := range strings.Split(strings.TrimPrefix(p.dir, current), string(filepath.Separator)) {
		current = filepath.Join(current, part)
		info, err := os.Lstat(current)
		created := false
		if errors.Is(err, os.ErrNotExist) && create {
			err = os.Mkdir(current, 0700)
			created = err == nil
			if err != nil && !errors.Is(err, os.ErrExist) {
				return err
			}
			info, err = os.Lstat(current)
		}
		if err != nil {
			return err
		}
		managed := current == p.root || strings.HasPrefix(current, p.root+string(filepath.Separator))
		if err := checkDirectory(info, managed || created); err != nil {
			return fmt.Errorf("%s: %w", current, err)
		}
		if err := checkPathACL(current); err != nil {
			return err
		}
		if created {
			if err := syncDirectory(filepath.Dir(current)); err != nil {
				return err
			}
		}
	}
	return nil
}

func checkDirectory(info os.FileInfo, private bool) error {
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%w: not a real directory", ErrUnsafe)
	}
	if private {
		return checkPrivate(info, 0700, false)
	}
	return checkAncestor(info)
}

func checkPrivate(info os.FileInfo, mode os.FileMode, regular bool) error {
	if info.Mode().Perm() != mode || info.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
		return fmt.Errorf("%w: require mode %04o", ErrUnsafe, mode)
	}
	if regular && !info.Mode().IsRegular() {
		return fmt.Errorf("%w: not a regular file", ErrUnsafe)
	}
	return checkOwner(info, regular)
}

// ReadFile opens a single protected file without following links and validates
// the opened inode before reading. Names must be single safe segments.
func (p Path) ReadFile(name string) ([]byte, error) {
	if err := segment(name); err != nil {
		return nil, err
	}
	// Coordinate with atomic replacement: after rename, an already-open old
	// inode has link count zero and must not be mistaken for corrupt state.
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
	return io.ReadAll(f)
}

func openProtected(path string) (*os.File, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if err := checkPrivate(info, 0600, true); err != nil {
		return nil, err
	}
	f, err := openNoFollow(path)
	if err != nil {
		return nil, err
	}
	actual, err := f.Stat()
	if err == nil {
		err = checkPrivate(actual, 0600, true)
	}
	if err == nil && !os.SameFile(info, actual) {
		err = fmt.Errorf("%w: file changed while opening", ErrUnsafe)
	}
	if err == nil {
		err = checkFileACL(f)
	}
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return f, nil
}

// Discover returns only an unambiguous existing directory. Empty applicable
// context fields match any safe segment. It fails closed on unsafe candidates;
// it never creates state, reads state files, or silently skips corrupt matches.
// An exact override bypasses enumeration, but not directory validation.
func Discover(o Options, c Context) (Path, error) {
	parts, err := c.parts()
	if err != nil {
		return Path{}, err
	}
	root, err := o.base()
	if err != nil {
		return Path{}, err
	}
	base := Path{root, root, c.Kind}
	if err := base.Validate(); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Path{}, ErrNotFound
		}
		return Path{}, err
	}
	if o.ExactDir != "" {
		return base, nil
	}
	var found []Path
	var visit func(Path, int) error
	visit = func(p Path, depth int) error {
		if depth == len(parts) {
			found = append(found, p)
			return nil
		}
		names := []string{parts[depth]}
		if names[0] == "" {
			entries, err := os.ReadDir(p.dir)
			if err != nil {
				return err
			}
			names = nil
			for _, entry := range entries {
				if strings.HasPrefix(entry.Name(), temporaryPrefix) {
					continue
				}
				if err := segment(entry.Name()); err != nil {
					return err
				}
				names = append(names, entry.Name())
			}
		}
		for _, name := range names {
			child := Path{filepath.Join(p.dir, name), root, c.Kind}
			if err := child.Validate(); err != nil {
				if errors.Is(err, os.ErrNotExist) {
					continue
				}
				return err
			}
			if err := visit(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := visit(base, 0); err != nil {
		return Path{}, err
	}
	switch len(found) {
	case 0:
		return Path{}, ErrNotFound
	case 1:
		return found[0], nil
	default:
		return Path{}, ErrAmbiguous
	}
}
