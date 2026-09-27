//go:build !darwin && !linux

package workercontrol

import (
	"errors"
	"testing"
)

func TestUnsupportedPeerInspection(t *testing.T) {
	peer, err := inspectPeerFD(0)
	if peer != (PeerIdentity{}) || !errors.Is(err, ErrPeerInspection) {
		t.Fatalf("unsupported platform returned peer=%+v err=%v", peer, err)
	}
}
