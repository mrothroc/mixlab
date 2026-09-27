package principal

import (
	"crypto"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

// SnapshotReceiver exposes an unexpired installed node identity only for a
// snapshot-recovery TLS listener. All application requests still need Active.
// Updating trust never renews an expired certificate or regenerates a key.
func (s *Store) SnapshotReceiver(now time.Time) (State, crypto.Signer, error) {
	_, r, _, err := s.load(now)
	if err != nil {
		return State{}, nil, err
	}
	a, err := trust.PinRoot(r.Root, s.pin, now)
	if err != nil {
		return State{}, nil, err
	}
	id, err := trust.VerifySnapshotReceiver(a, r.Chain, now)
	if err != nil {
		return State{}, nil, err
	}
	if id != r.Principal {
		return State{}, nil, fmt.Errorf("snapshot receiver identity changed")
	}
	k, err := s.keys.Signer(r.Key)
	return r, k, err
}
