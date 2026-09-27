package trust

import (
	"bytes"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

const MaxWorkloadLifetime = certificates.WorkloadLifetime

// VerifyWorkload authenticates an exact admitted DDP workload, including rank
// and attempt. The caller supplies want from trusted admission state, not from
// the peer. This is not a membership or job authorization decision.
func VerifyWorkload(a Anchor, v VerifiedSnapshot, chain [][]byte, want WorkloadBinding, now time.Time) error {
	if err := v.fresh(now); err != nil {
		return err
	}
	id, err := principal(a, chain, v, now)
	if err != nil {
		return err
	}
	if id.Role != Worker || id.Workload == nil || *id.Workload != want {
		return fmt.Errorf("workload does not match admitted binding")
	}
	return v.checkRevocation(id, 0)
}

// NodeEnvelopeKey returns only an authenticated, currently eligible node key.
// Public bytes are copied; registration/renewal remain authority-owned actions.
func NodeEnvelopeKey(a Anchor, v VerifiedSnapshot, chain [][]byte, wantNode string, now time.Time) ([]byte, error) {
	if err := v.fresh(now); err != nil {
		return nil, err
	}
	id, err := principal(a, chain, v, now)
	if err != nil {
		return nil, err
	}
	if !certificates.ValidID(wantNode) || id.Role != Node || id.Principal != wantNode || len(id.EnvelopeKey) != 32 {
		return nil, fmt.Errorf("envelope recipient is not the intended enrolled node")
	}
	if err := v.checkRevocation(id, 0); err != nil {
		return nil, err
	}
	return bytes.Clone(id.EnvelopeKey), nil
}
