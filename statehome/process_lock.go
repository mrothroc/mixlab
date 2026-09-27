package statehome

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// WithProcessLock serializes an owning lifecycle across processes while allowing
// normal protected-file operations inside fn. The lock file is permanent: never
// replace or unlink it while the context exists. Crashes release the OS lock.
// This is not a transaction; fn must journal its own external side effects.
func (p Path) WithProcessLock(ctx context.Context, name string, fn func() error) error {
	if ctx == nil || fn == nil {
		return fmt.Errorf("missing lock context or operation")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := segment(name); err != nil {
		return err
	}
	marker := []byte("mixlab_process_lock_v1\n")
	if err := p.CompareAndSwap(name, nil, marker); err != nil && !errors.Is(err, ErrConflict) {
		return err
	}
	f, err := openProtected(filepath.Join(p.dir, name))
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	for {
		locked, err := tryProcessLock(f)
		if err != nil {
			return err
		}
		if locked {
			break
		}
		timer := time.NewTimer(10 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	defer unlockDirectory(f)
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := p.Validate(); err != nil {
		return err
	}
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	named, err := os.Lstat(filepath.Join(p.dir, name))
	if err != nil {
		return err
	}
	if !os.SameFile(opened, named) {
		return fmt.Errorf("%w: lifecycle lock replaced", ErrUnsafe)
	}
	b, err := io.ReadAll(io.LimitReader(f, int64(len(marker)+1)))
	if err != nil {
		return err
	}
	if !bytes.Equal(b, marker) {
		return fmt.Errorf("invalid lifecycle lock marker")
	}
	return fn()
}
