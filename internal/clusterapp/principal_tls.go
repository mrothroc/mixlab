package clusterapp

import (
	"fmt"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/principal"
)

// A connection snapshots local credentials, but peer verification always reads
// current protected trust. Nothing in a discovery response can change the pin.
func managedPrincipalPolicy(p *principal.Store, role trust.Role, clock func() time.Time) (*managedtls.Policy, error) {
	return managedPrincipalPolicyRoles(p, clock, role)
}
func managedPrincipalPolicyRoles(p *principal.Store, clock func() time.Time, roles ...trust.Role) (*managedtls.Policy, error) {
	if p == nil || clock == nil {
		return nil, fmt.Errorf("principal and clock required")
	}
	s, key, err := p.Active(clock())
	if err != nil {
		return nil, err
	}
	a, err := trust.PinRoot(s.Root, s.Fingerprint, clock())
	if err != nil {
		return nil, err
	}
	return managedtls.New(managedtls.Identity{Chain: s.Chain, Key: key}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
		current, err := p.View(now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		v, err := trust.VerifySnapshot(a, current.Snapshot, now)
		if err != nil {
			return trust.AuthenticatedPrincipal{}, err
		}
		peer, err := trust.AuthenticatePrincipal(a, v, chain, now)
		if err == nil && !slices.Contains(roles, peer.Role) {
			err = fmt.Errorf("unexpected managed peer role")
		}
		return peer, err
	}, clock)
}
