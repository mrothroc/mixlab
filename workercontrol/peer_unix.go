//go:build darwin || linux

package workercontrol

import "syscall"

func requireUnixStream(fd int) error {
	kind, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_TYPE)
	if err != nil {
		return err
	}
	if kind != syscall.SOCK_STREAM {
		return ErrPeerInspection
	}
	addr, err := syscall.Getpeername(fd)
	if err != nil {
		return err
	}
	if _, ok := addr.(*syscall.SockaddrUnix); !ok {
		return ErrPeerInspection
	}
	return nil
}
