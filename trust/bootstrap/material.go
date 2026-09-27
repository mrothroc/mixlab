package bootstrap

import (
	"bytes"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

// AuthorityMaterial is the completed bootstrap handoff to trusted composition.
// It contains public certificates and protected handles, never the root signing
// handle. Root-key access remains an explicit initialization/rekey operation.
type AuthorityMaterial struct {
	Anchor                 trust.Anchor
	Issuer, SnapshotSigner []byte
	IssuerKey, SnapshotKey securekeys.Handle
	InitialSnapshot        trust.SignedSnapshot
	AuthorityPrincipal     statehome.Path
}

func LoadAuthority(p statehome.Path, now time.Time) (AuthorityMaterial, error) {
	var out AuthorityMaterial
	if p.Kind() != statehome.Authority {
		return out, fmt.Errorf("authority context required")
	}
	x := &runner{path: p, store: p, now: now}
	if err := x.read(); err != nil {
		return out, err
	}
	if !x.r.Ready || x.r.Snapshot == nil {
		return out, fmt.Errorf("initialization incomplete; explicit init recovery required")
	}
	a, err := x.anchor()
	if err != nil {
		return out, err
	}
	b, err := encode(x.r.Snapshot)
	if err != nil {
		return out, err
	}
	if _, err := trust.RestoreSnapshot(a, b, now); err != nil {
		return out, err
	}
	if err := checkContext(p, x.r, x.r.Contexts[0]); err != nil {
		return out, err
	}
	principal, err := resolved(x.r.Contexts[1].Final, statehome.Principal)
	if err != nil {
		return out, err
	}
	c := x.r.Contexts[0]
	return AuthorityMaterial{Anchor: a, Issuer: bytes.Clone(c.Keys[1].Certificate), SnapshotSigner: bytes.Clone(c.Keys[2].Certificate), IssuerKey: *c.Keys[1].Handle, SnapshotKey: *c.Keys[2].Handle, InitialSnapshot: *x.r.Snapshot, AuthorityPrincipal: principal}, nil
}
