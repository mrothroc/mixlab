//go:build darwin || linux

package local

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/statehome"
)

func listenShort(limits Limits) (*Server, *os.File, error) {
	// Do not use inherited TMPDIR: the supervisor intentionally sets that to the
	// (possibly long) persistent attempt directory. MkdirTemp creates mode 0700.
	base, err := filepath.EvalSymlinks("/tmp")
	if err != nil {
		return nil, nil, err
	}
	if len(base) > 60 {
		return nil, nil, fmt.Errorf("system temporary path too long for local IPC")
	}
	path, err := os.MkdirTemp(base, "mwc-")
	if err != nil {
		return nil, nil, err
	}
	keep := false
	defer func() {
		if !keep {
			_ = os.Remove(path)
		}
	}()
	info, err := os.Lstat(path)
	if err != nil {
		return nil, nil, err
	}
	dir, err := statehome.Resolve(statehome.Options{ExactDir: path}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return nil, nil, err
	}
	s, credential, err := Listen(dir, limits)
	if err != nil {
		return nil, nil, err
	}
	s.transient, s.transientInode = path, info
	keep = true
	return s, credential, nil
}
