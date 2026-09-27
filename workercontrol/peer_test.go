package workercontrol

import (
	"errors"
	"syscall"
	"testing"
)

type failedRawConn struct{ err error }

func (r failedRawConn) Control(func(uintptr)) error    { return r.err }
func (r failedRawConn) Read(func(uintptr) bool) error  { panic("unexpected Read") }
func (r failedRawConn) Write(func(uintptr) bool) error { panic("unexpected Write") }

func TestPeerInspectionFailsClosed(t *testing.T) {
	want := errors.New("closed connection")
	for _, raw := range []syscall.RawConn{nil, failedRawConn{want}, failedRawConn{}} {
		peer, err := InspectPeerIdentity(raw)
		if !errors.Is(err, ErrPeerInspection) || peer != (PeerIdentity{}) {
			t.Fatalf("peer=%+v err=%v", peer, err)
		}
	}
	if _, err := InspectPeerIdentity(failedRawConn{want}); !errors.Is(err, want) {
		t.Fatal(err)
	}
}
