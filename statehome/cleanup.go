package statehome

import (
	"os"
	"path/filepath"
	"strings"
)

// CleanupTemporaryFiles removes abandoned atomic-file staging in this exact
// directory. The directory lock excludes all cooperating live file publishers.
// Staging directories are intentionally untouched: they have a separate owner.
func (p Path) CleanupTemporaryFiles() (int, error) {
	dir, err := p.lock()
	if err != nil {
		return 0, err
	}
	defer release(dir)
	entries, err := os.ReadDir(p.dir)
	if err != nil {
		return 0, err
	}
	removed := 0
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), temporaryPrefix) || entry.IsDir() {
			continue
		}
		path := filepath.Join(p.dir, entry.Name())
		f, err := openProtected(path)
		if err != nil {
			return removed, err
		}
		if err = f.Close(); err != nil {
			return removed, err
		}
		if err = p.sameDirectory(dir); err != nil {
			return removed, err
		}
		if err = os.Remove(path); err != nil {
			return removed, err
		}
		removed++
	}
	if removed > 0 {
		return removed, dir.Sync()
	}
	return 0, nil
}
