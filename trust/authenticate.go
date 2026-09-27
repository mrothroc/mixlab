package trust

import (
	"bytes"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

// ProposedEnrollmentRoot validates a source's self-consistent certificate
// profile, not its identity. Its result must be accepted by explicit trusted-LAN
// policy or human root comparison before any key/request is created.
func ProposedEnrollmentRoot(chain [][]byte, role Role, now time.Time) (Anchor, error) {
	if len(chain) != 3 || len(chain[2]) > 16<<10 {
		return Anchor{}, fmt.Errorf("invalid proposed enrollment chain")
	}
	c, _, err := certificates.Parse(chain[2])
	if err != nil {
		return Anchor{}, err
	}
	fingerprint, err := RootFingerprint(c.PublicKey)
	if err != nil {
		return Anchor{}, err
	}
	a, err := PinRoot(chain[2], fingerprint, now)
	if err != nil {
		return Anchor{}, err
	}
	if err := VerifyProvisionalSource(a, chain, role, now); err != nil {
		return Anchor{}, err
	}
	return a, nil
}

// VerifyProvisionalSource is only a pinned-root enrollment handshake check.
// It deliberately returns no AuthenticatedPrincipal. Before sending a one-use
// secret or request, the client must obtain a fresh signed snapshot and call
// AuthenticatePrincipal on this same TLS peer. No normal operation may use this
// check as a substitute for current revocation/role eligibility checks.
func VerifyProvisionalSource(a Anchor, chain [][]byte, role Role, now time.Time) error {
	if (role != Authority && role != Coordinator) || len(chain) != 3 || !bytes.Equal(chain[2], a.DER()) {
		return fmt.Errorf("invalid provisional enrollment source")
	}
	for _, der := range chain {
		if len(der) == 0 || len(der) > 16<<10 {
			return fmt.Errorf("invalid enrollment source certificate size")
		}
	}
	_, id, err := certificates.Verify(a, chain[0], chain[1], certificates.Principal, now)
	if err != nil {
		return err
	}
	if id.Role != role {
		return fmt.Errorf("wrong provisional source role")
	}
	return nil
}

// AuthenticatedPrincipal is identity evidence, not permission to perform a node
// operation. Callers must still apply their own role and admission policies.
type AuthenticatedPrincipal struct {
	Cluster, Principal, Serial string
	Role                       Role
	NotBefore, ExpiresAt       time.Time
}

// AuthenticatePrincipal checks current trust, the exact profile and issuer,
// role eligibility, and revocation. TLS adapters must also verify possession of
// the certificate key through the handshake; a supplied chain alone is not PoP.
func AuthenticatePrincipal(a Anchor, v VerifiedSnapshot, chain [][]byte, now time.Time) (AuthenticatedPrincipal, error) {
	if err := v.fresh(now); err != nil {
		return AuthenticatedPrincipal{}, err
	}
	id, err := principal(a, chain, v, now)
	if err != nil {
		return AuthenticatedPrincipal{}, err
	}
	if err := v.checkRevocation(id, 0); err != nil {
		return AuthenticatedPrincipal{}, err
	}
	c, _, err := certificates.Parse(chain[0])
	if err != nil {
		return AuthenticatedPrincipal{}, err
	}
	return AuthenticatedPrincipal{a.Cluster(), id.Principal, id.Serial, id.Role, c.NotBefore, c.NotAfter}, nil
}
