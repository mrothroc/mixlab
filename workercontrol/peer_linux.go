package workercontrol

import "syscall"

func inspectPeerFD(fd int) (PeerIdentity, error) {
	if err := requireUnixStream(fd); err != nil {
		return PeerIdentity{}, err
	}
	cred, err := syscall.GetsockoptUcred(fd, syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	if err != nil {
		return PeerIdentity{}, err
	}
	if cred.Pid <= 0 {
		return PeerIdentity{}, ErrPeerInspection
	}
	return PeerIdentity{PID: int(cred.Pid), UID: cred.Uid}, nil
}
