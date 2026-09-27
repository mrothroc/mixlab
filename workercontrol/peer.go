package workercontrol

import (
	"errors"
	"fmt"
	"syscall"
)

var ErrPeerInspection = errors.New("workercontrol: OS peer inspection failed")

// InspectPeerIdentity reads kernel credentials from a connected Unix stream
// socket while RawConn.Control keeps its descriptor alive. It neither opens
// nor owns the connection. Only Darwin/Linux are supported; other platforms
// fail closed. Pass a RawConn obtained directly from the accepted connection,
// not an untrusted implementation or message-supplied descriptor.
//
// Credentials identify the peer at connection establishment, not its current
// executable, liveness, or a later recipient of a passed socket descriptor.
// Compare them to the supervised child via ValidatePeerIdentity/Authenticate;
// the host must prevent PID reuse and close sessions on attempt termination.
func InspectPeerIdentity(raw syscall.RawConn) (PeerIdentity, error) {
	if raw == nil {
		return PeerIdentity{}, ErrPeerInspection
	}
	var peer PeerIdentity
	var inspectErr error
	called := false
	err := raw.Control(func(fd uintptr) {
		called = true
		peer, inspectErr = inspectPeerFD(int(fd))
	})
	if err != nil {
		return PeerIdentity{}, fmt.Errorf("%w: %w", ErrPeerInspection, err)
	}
	if inspectErr != nil {
		return PeerIdentity{}, fmt.Errorf("%w: %w", ErrPeerInspection, inspectErr)
	}
	if !called || peer.PID <= 0 {
		return PeerIdentity{}, ErrPeerInspection
	}
	return peer, nil
}
