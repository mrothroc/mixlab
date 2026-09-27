//go:build !darwin && !linux

package workercontrol

func inspectPeerFD(int) (PeerIdentity, error) {
	return PeerIdentity{}, ErrPeerInspection
}
