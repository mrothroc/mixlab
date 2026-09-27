package principal

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

// Install publishes into the enrollee's pre-existing private staging directory.
// The caller must validate its policy-specific enrollment result first, then
// promote this complete staging context atomically to the absent final path.
// Keys are created by keylifecycle before enrollment, never by this function.
func Install(ctx context.Context, stage statehome.Path, acceptedRoot trust.Anchor, state State, now time.Time) error {
	if stage.Kind() != statehome.Enrollment || state.Version != Version || state.Fingerprint != acceptedRoot.Fingerprint() || state.Cluster != acceptedRoot.Cluster() {
		return fmt.Errorf("installation differs from accepted root or staging context")
	}
	b, err := json.Marshal(state)
	if err != nil {
		return err
	}
	state, err = decode(b)
	if err != nil {
		return err
	}
	keys, err := securekeys.OpenSelected(state.Key.Backend, stage, state.Key.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = keys.Close() }()
	s := &Store{path: stage, keys: keys, pin: acceptedRoot.Fingerprint(), principal: state.Principal, role: state.Role}
	a, v, err := s.validate(state, now)
	if err != nil {
		return err
	}
	if _, err := trust.AuthenticatePrincipal(a, v, state.Chain, now); err != nil {
		return err
	}
	return stage.WithProcessLock(ctx, lockname, func() error { return s.save(nil, state) })
}
