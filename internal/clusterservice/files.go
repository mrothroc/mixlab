package clusterservice

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/mrothroc/mixlab/statehome"
)

// Unit directories may already be 0755 (unlike private Mixlab state). Inspect
// every component and never follow symlinks or replace an unrelated definition.
func safeUnitParent(path string) error {
	dir := filepath.Dir(path)
	if dir != path {
		if err := safeUnitParent(dir); err != nil {
			return err
		}
	}
	i, err := os.Lstat(path)
	if os.IsNotExist(err) {
		if err = os.Mkdir(path, 0700); err != nil {
			return err
		}
		i, err = os.Lstat(path)
	}
	if err != nil {
		return err
	}
	if !i.IsDir() || i.Mode()&os.ModeSymlink != 0 || i.Mode().Perm()&0022 != 0 {
		// The sticky system temp directory is allowed only as an ancestor of
		// private test paths, never as the service definition directory.
		if !i.IsDir() || i.Mode()&os.ModeSticky == 0 {
			return fmt.Errorf("unsafe service directory %s", path)
		}
	}
	return nil
}

func checkUnit(path string, expected []byte) error {
	if err := safeUnitParent(filepath.Dir(path)); err != nil {
		return err
	}
	i, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !i.Mode().IsRegular() || i.Mode().Perm() != 0600 {
		return fmt.Errorf("unsafe service definition")
	}
	if st, ok := i.Sys().(*syscall.Stat_t); !ok || st.Uid != uint32(os.Getuid()) {
		return fmt.Errorf("service definition is not owned by this user")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if !bytes.Equal(b, expected) {
		return fmt.Errorf("service definition differs; refusing to replace/remove it")
	}
	return nil
}

func publishUnit(path string, b []byte) error {
	if err := safeUnitParent(filepath.Dir(path)); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".mixlab-unit-")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(f.Name()) }()
	_, writeErr := f.Write(b)
	if writeErr == nil {
		writeErr = f.Sync()
	}
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	// Publish the complete definition without replacing an existing inode.
	if err := os.Link(f.Name(), path); err != nil {
		if os.IsExist(err) {
			return checkUnit(path, b)
		}
		return err
	}
	return nil
}

// TailLog bounds retained service output even during a persistent restart loop.
// Files remain in private state and survive service uninstall for diagnosis.
type TailLog struct {
	mu   sync.Mutex
	Path statehome.Path
}

func (l *TailLog) Write(b []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	const limit = 256 << 10
	old, err := l.Path.ReadFileLimit("service.log", limit)
	if err != nil && !os.IsNotExist(err) {
		return 0, err
	}
	old = append(old, b...)
	tail := old
	if len(tail) > limit {
		tail = tail[len(tail)-limit:]
	}
	if err := l.Path.WriteFile("service.log", tail); err != nil {
		return 0, err
	}
	return len(b), nil
}
