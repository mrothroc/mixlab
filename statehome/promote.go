package statehome

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Promote atomically moves a durable, fully populated staging context to this
// absent destination. It never copies across filesystems or replaces existing
// state. The owner must stop context writers and hold its lifecycle lock before
// calling. An error after rename can mean publication succeeded: reconcile the
// exact destination, never recreate staging keys.
func (p Path) Promote(stage Path) error {
	if p.dir == "" || stage.dir == "" || p.dir == stage.dir ||
		strings.HasPrefix(p.dir, stage.dir+string(filepath.Separator)) || strings.HasPrefix(stage.dir, p.dir+string(filepath.Separator)) {
		return fmt.Errorf("%w: overlapping promotion contexts", ErrUnsafe)
	}
	parent := Path{filepath.Dir(p.dir), p.root, p.kind}
	if err := parent.Ensure(); err != nil {
		return err
	}
	// Order directory locks independently of source/destination roles.
	first, second := stage, parent
	if first.dir > second.dir {
		first, second = second, first
	}
	a, err := first.lock()
	if err != nil {
		return err
	}
	defer release(a)
	b, err := second.lock()
	if err != nil {
		return err
	}
	defer release(b)
	if err := absent(p.dir); err != nil {
		return err
	}
	if err := syncTree(stage); err != nil {
		return err
	}
	if err := first.sameDirectory(a); err != nil {
		return err
	}
	if err := second.sameDirectory(b); err != nil {
		return err
	}
	if err := absent(p.dir); err != nil {
		return err
	}
	if err := os.Rename(stage.dir, p.dir); err != nil {
		return err
	}
	return errors.Join(syncDirectory(parent.dir), syncDirectory(filepath.Dir(stage.dir)))
}
