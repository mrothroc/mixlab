//go:build darwin

package train

import "golang.org/x/sys/unix"

func publishGridDirectory(from, to string) error {
	return unix.RenameatxNp(unix.AT_FDCWD, from, unix.AT_FDCWD, to, unix.RENAME_EXCL)
}
