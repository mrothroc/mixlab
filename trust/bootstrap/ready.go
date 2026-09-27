package bootstrap

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/principal"
)

// Once bootstrap is complete, installed credentials own refresh and renewal.
// Recovery validates those identities instead of restoring their original leaf
// or snapshot bytes, which would roll back legitimate lifecycle transitions.
func (x *runner) verifyReady() error {
	a, err := x.anchor()
	if err != nil {
		return err
	}
	ca := x.r.Contexts[0]
	if err := checkContext(x.path, x.r, ca); err != nil {
		return err
	}
	keys, err := securekeys.OpenSelected(x.r.Backend, x.path, ca.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = keys.Close() }()
	m, err := keylifecycle.Open(x.path, keys, ca.Owner)
	if err != nil {
		return err
	}
	for _, k := range ca.Keys {
		r, err := m.View(k.Slot)
		if err != nil {
			return err
		}
		got, _ := json.Marshal(r.Active)
		want, _ := json.Marshal(k.Handle)
		if r.Stage != "active" || !bytes.Equal(got, want) {
			return fmt.Errorf("CA key lifecycle changed; explicit rekey required")
		}
		if _, err := keys.Signer(*k.Handle); err != nil {
			return err
		}
	}
	for _, c := range x.r.Contexts[1:] {
		p, err := selectedPath(c)
		if err != nil {
			return err
		}
		if err := checkContext(p, x.r, c); err != nil {
			return err
		}
		s, err := principal.Open(p, x.now)
		if err != nil {
			return err
		}
		r, err := s.View(x.now)
		closeErr := s.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		if r.Cluster != a.Cluster() || r.Fingerprint != a.Fingerprint() || r.Principal != c.Principal || string(r.Role) != c.Owner {
			return fmt.Errorf("initial principal identity changed")
		}
	}
	return nil
}
