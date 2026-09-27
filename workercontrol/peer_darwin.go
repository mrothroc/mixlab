package workercontrol

import (
	"syscall"
	"unsafe"
)

// Darwin sys/un.h and sys/ucred.h: LOCAL_PEERCRED, LOCAL_PEERPID, xucred v0.
// Go's frozen syscall API has no getpeereid wrapper. Read the same
// LOCAL_PEERCRED option as Apple's getpeereid implementation, without cgo:
// https://github.com/apple-oss-distributions/Libc/blob/main/gen/FreeBSD/getpeereid.c
const (
	localPeerCred = 1
	localPeerPID  = 2
)

type darwinXucred struct {
	Version uint32
	UID     uint32
	NGroups int16
	_       [2]byte
	Groups  [16]uint32
}

func inspectPeerFD(fd int) (PeerIdentity, error) {
	if err := requireUnixStream(fd); err != nil {
		return PeerIdentity{}, err
	}
	pid, err := syscall.GetsockoptInt(fd, 0, localPeerPID)
	if err != nil {
		return PeerIdentity{}, err
	}
	var cred darwinXucred
	size := uint32(unsafe.Sizeof(cred))
	_, _, errno := syscall.Syscall6(syscall.SYS_GETSOCKOPT, uintptr(fd), 0, localPeerCred,
		uintptr(unsafe.Pointer(&cred)), uintptr(unsafe.Pointer(&size)), 0)
	if errno != 0 {
		return PeerIdentity{}, errno
	}
	if size != uint32(unsafe.Sizeof(cred)) || cred.Version != 0 || pid <= 0 {
		return PeerIdentity{}, ErrPeerInspection
	}
	return PeerIdentity{PID: pid, UID: cred.UID}, nil
}
