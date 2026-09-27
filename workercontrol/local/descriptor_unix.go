//go:build darwin || linux

package local

import (
	"os"
	"syscall"
)

// ExtraFiles arrives as a blocking descriptor after exec. Reopen a duplicate
// through Go's poller so deadline/cancellation works even when the worker used
// os.NewFile on FD3. Never change the process-global umask or expose a disk file.
func pollablePipe(original *os.File) (*os.File, error) {
	raw, err := original.SyscallConn()
	if err != nil {
		return nil, err
	}
	fd := -1
	var setupErr error
	err = raw.Control(func(value uintptr) {
		syscall.ForkLock.RLock()
		defer syscall.ForkLock.RUnlock()
		fd, setupErr = syscall.Dup(int(value))
		if setupErr == nil {
			syscall.CloseOnExec(fd)
			setupErr = syscall.SetNonblock(fd, true)
		}
	})
	if err != nil || setupErr != nil {
		if fd >= 0 {
			_ = syscall.Close(fd)
		}
		if err != nil {
			return nil, err
		}
		return nil, setupErr
	}
	return os.NewFile(uintptr(fd), "worker-credential"), nil
}
