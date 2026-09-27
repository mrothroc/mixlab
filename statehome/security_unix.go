//go:build darwin || linux

package statehome

import (
	"fmt"
	"os"
	"syscall"
)

func checkOwner(info os.FileInfo, regular bool) error {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || stat.Uid != uint32(os.Geteuid()) {
		return fmt.Errorf("%w: require effective-user ownership", ErrUnsafe)
	}
	if regular && stat.Nlink != 1 {
		return fmt.Errorf("%w: protected file has multiple hard links", ErrUnsafe)
	}
	return nil
}

func checkAncestor(info os.FileInfo) error {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || (stat.Uid != 0 && stat.Uid != uint32(os.Geteuid())) {
		return fmt.Errorf("%w: untrusted ancestor owner", ErrUnsafe)
	}
	// Root-owned sticky temporary directories are safe parents of private state.
	if info.Mode().Perm()&0022 != 0 && (stat.Uid != 0 || info.Mode()&os.ModeSticky == 0) {
		return fmt.Errorf("%w: writable ancestor", ErrUnsafe)
	}
	return nil
}

func openNoFollow(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
}

func lockDirectory(f *os.File) error {
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
		if err != syscall.EINTR {
			return err
		}
	}
}

func unlockDirectory(f *os.File) { _ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }

func tryProcessLock(f *os.File) (bool, error) {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err == syscall.EWOULDBLOCK || err == syscall.EAGAIN || err == syscall.EINTR {
		return false, nil
	}
	return err == nil, err
}
