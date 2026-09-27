package bootstrap

import (
	"bytes"
	"context"
	"crypto"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
)

func (x *runner) run(ctx context.Context) error {
	for n := range x.r.Contexts {
		p, err := x.context(ctx, n)
		if err != nil {
			return err
		}
		if err := x.keys(ctx, n, p); err != nil {
			return err
		}
	}
	if err := x.certificates(); err != nil {
		return err
	}
	if err := x.publicTrust(); err != nil {
		return err
	}
	for n := 1; n < len(x.r.Contexts); n++ {
		p, err := x.context(ctx, n)
		if err != nil {
			return err
		}
		if !x.r.Contexts[n].Published {
			want, err := x.principalBytes(n)
			if err != nil {
				return err
			}
			old, err := p.ReadFileLimit(credentialFile, maxBytes)
			switch {
			case errors.Is(err, os.ErrNotExist):
				if err := p.CompareAndSwap(credentialFile, nil, want); err != nil {
					return err
				}
			case err != nil:
				return err
			case !bytes.Equal(old, want):
				return fmt.Errorf("staged principal credentials differ from initialization plan")
			}
			if err := x.verifyPrincipal(p, n); err != nil {
				return err
			}
			final, err := resolved(x.r.Contexts[n].Final, statehome.Principal)
			if err != nil {
				return err
			}
			if err := final.Promote(p); err != nil {
				return err
			}
			x.r.Contexts[n].Published = true
			if err := x.save(); err != nil {
				return err
			}
			p = final
		}
		if err := x.verifyPrincipal(p, n); err != nil {
			return err
		}
	}
	if !x.r.Ready {
		x.r.Contexts[0].Published, x.r.Ready = true, true
		return x.save()
	}
	return nil
}

func (x *runner) publicTrust() error {
	a, err := x.anchor()
	if err != nil {
		return err
	}
	if x.r.Endpoints == nil {
		err = x.caSigner(0, func(key crypto.Signer) error {
			e, err := trust.SignAuthorityEndpoints(a, trust.AuthorityEndpoints{
				Version: trust.EndpointVersion, Cluster: x.r.Cluster, Audience: x.r.Audience,
				URLs: []string{x.r.Endpoint}, IssuedAt: x.now.Unix(), ExpiresAt: x.now.Add(180 * 24 * time.Hour).Unix(),
			}, key, x.now)
			if err == nil {
				x.r.Endpoints = &e
			}
			return err
		})
		if err != nil {
			return err
		}
		if err := x.save(); err != nil {
			return err
		}
	}
	if x.r.Snapshot == nil {
		err = x.caSigner(2, func(key crypto.Signer) error {
			s, err := trust.SignSnapshot(a, trust.Snapshot{
				Version: trust.SnapshotVersion, Cluster: x.r.Cluster, Generation: 1,
				IssuedAt: x.now.Unix(), ExpiresAt: x.now.Add(trust.SnapshotLifetime).Unix(),
				Issuers:       [][]byte{x.r.Contexts[0].Keys[1].Certificate},
				EligibleRoles: []trust.Role{trust.Authority, trust.Controller, trust.Coordinator, trust.Node, trust.Worker},
				Endpoints:     *x.r.Endpoints,
			}, x.r.Contexts[0].Keys[2].Certificate, key, x.now)
			if err == nil {
				x.r.Snapshot = &s
			}
			return err
		})
		if err != nil {
			return err
		}
		if err := x.save(); err != nil {
			return err
		}
	}
	// Recovery verifies the original signed snapshot without refreshing or
	// treating its expired authorization window as current trust.
	b, err := encode(x.r.Snapshot)
	if err != nil {
		return err
	}
	if _, err := trust.RestoreSnapshot(a, b, x.now); err != nil {
		return err
	}
	want, _ := encode(x.r.Endpoints)
	actual, _ := encode(x.r.Snapshot.Payload.Endpoints)
	if !bytes.Equal(want, actual) || x.r.Snapshot.Payload.Generation != 1 ||
		!bytes.Equal(x.r.Snapshot.SignerCertificate, x.r.Contexts[0].Keys[2].Certificate) ||
		len(x.r.Snapshot.Payload.Issuers) != 1 || !bytes.Equal(x.r.Snapshot.Payload.Issuers[0], x.r.Contexts[0].Keys[1].Certificate) {
		return fmt.Errorf("bootstrap trust differs from initialization plan")
	}
	return nil
}

func (x *runner) principalBytes(n int) ([]byte, error) {
	if x.r.Snapshot == nil {
		return nil, fmt.Errorf("missing bootstrap trust")
	}
	a, err := x.anchor()
	if err != nil {
		return nil, err
	}
	c := x.r.Contexts[n]
	if c.Keys[0].Handle == nil {
		return nil, fmt.Errorf("missing principal key")
	}
	return encode(PrincipalState{
		Version: version, Cluster: x.r.Cluster, Role: trust.Role(c.Owner), Principal: c.Principal,
		Key: *c.Keys[0].Handle, Root: a.DER(), Fingerprint: a.Fingerprint(),
		Chain: [][]byte{c.Keys[0].Certificate, x.r.Contexts[0].Keys[1].Certificate, a.DER()}, Snapshot: *x.r.Snapshot,
	})
}

func (x *runner) verifyPrincipal(p statehome.Path, n int) error {
	c := x.r.Contexts[n]
	if err := checkContext(p, x.r, c); err != nil {
		return err
	}
	want, err := x.principalBytes(n)
	if err != nil {
		return err
	}
	b, err := p.ReadFileLimit(credentialFile, maxBytes)
	if err != nil {
		return err
	}
	if !bytes.Equal(want, b) {
		return fmt.Errorf("published principal differs from initialization plan")
	}
	s, err := securekeys.OpenSelected(x.r.Backend, p, c.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = s.Close() }()
	m, err := keylifecycle.Open(p, s, c.Owner)
	if err != nil {
		return err
	}
	r, err := m.View(keylifecycle.Principal)
	if err != nil {
		return err
	}
	actual, _ := encode(r.Active)
	expected, _ := encode(c.Keys[0].Handle)
	if r.Stage != "active" || !bytes.Equal(actual, expected) {
		return fmt.Errorf("principal key lifecycle differs from initialization plan")
	}
	_, err = s.Signer(*c.Keys[0].Handle)
	if err != nil {
		return err
	}
	a, err := x.anchor()
	if err != nil {
		return err
	}
	snapshot, err := encode(x.r.Snapshot)
	if err != nil {
		return err
	}
	v, err := trust.RestoreSnapshot(a, snapshot, x.now)
	if err != nil {
		return err
	}
	// This verifies issuance, not present-day authorization. Future services
	// must refresh trust and authenticate against their current clock.
	_, err = trust.AuthenticatePrincipal(a, v, [][]byte{c.Keys[0].Certificate, x.r.Contexts[0].Keys[1].Certificate, a.DER()}, time.Unix(x.r.Snapshot.Payload.IssuedAt, 0))
	return err
}
