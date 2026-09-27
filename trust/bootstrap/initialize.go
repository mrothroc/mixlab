package bootstrap

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/identity"
	"github.com/mrothroc/mixlab/trust/internal/certificates"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
)

type journal interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
	WithProcessLock(context.Context, string, func() error) error
}
type runner struct {
	path  statehome.Path
	store journal
	raw   []byte
	r     record
	now   time.Time
}

func Initialize(ctx context.Context, c Config, now time.Time) (Report, error) {
	base := Report{Cluster: c.Cluster, AuthorityDir: c.Authority.Dir()}
	if err := prepare(ctx, c, now); err != nil {
		return base, err
	}
	return Recover(ctx, c.Authority, now)
}

func prepare(ctx context.Context, c Config, now time.Time) error {
	if c.Authority.Kind() != statehome.Authority || len(c.Principals) != 3 {
		return fmt.Errorf("authority and three separate principal targets required")
	}
	backend, err := securekeys.ResolveBackend(c.Backend)
	if err != nil {
		return err
	}
	r := record{Version: version, Cluster: c.Cluster, Created: now.Unix(), Backend: backend, Endpoint: c.Endpoint, Audience: c.Audience}
	r.Contexts = append(r.Contexts, contextRecord{Owner: "cluster-authority", Final: c.Authority.Dir(), Staging: c.Authority.Dir(), Scope: scope(c.Cluster, "cluster-authority", ""),
		Keys: []keyRecord{{Slot: keylifecycle.Root}, {Slot: keylifecycle.Issuer}, {Slot: keylifecycle.SnapshotSigner}}})
	for _, p := range c.Principals {
		if p.Final.Kind() != statehome.Principal || p.Staging.Kind() != statehome.Enrollment {
			return fmt.Errorf("invalid principal staging context")
		}
		r.Contexts = append(r.Contexts, contextRecord{Owner: string(p.Role), Principal: p.Principal, Final: p.Final.Dir(), Staging: p.Staging.Dir(), Scope: scope(c.Cluster, string(p.Role), p.Principal), Keys: []keyRecord{{Slot: keylifecycle.Principal}}})
	}
	if err := r.validate(c.Authority.Dir()); err != nil {
		return err
	}
	for _, p := range c.Principals {
		for _, v := range []statehome.Path{p.Final, p.Staging} {
			if _, err := os.Lstat(v.Dir()); !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("bootstrap destination must be absent: %s", v.Dir())
			}
		}
	}
	return c.Authority.Publish(func(stage statehome.Path) error {
		if err := initializeContext(ctx, stage, r, r.Contexts[0]); err != nil {
			return err
		}
		b, err := encode(r)
		if err != nil {
			return err
		}
		return stage.CompareAndSwap(stateFile, nil, b)
	})
}

// Recover uses the persisted paths, backend, IDs and key intents. It never
// reinterprets new init options or regenerates a missing previously begun key.
func Recover(ctx context.Context, p statehome.Path, now time.Time) (Report, error) {
	if p.Kind() != statehome.Authority {
		return Report{}, fmt.Errorf("bootstrap requires authority context")
	}
	x := &runner{path: p, store: p, now: now}
	var report Report
	err := p.WithProcessLock(ctx, lockFile, func() error {
		if err := x.read(); err != nil {
			return err
		}
		var verifyErr error
		if x.r.Ready {
			verifyErr = x.verifyReady()
		} else {
			verifyErr = x.run(ctx)
		}
		if verifyErr != nil {
			return verifyErr
		}
		var err error
		report, err = x.report()
		return err
	})
	return report, err
}

func initializeContext(ctx context.Context, p statehome.Path, r record, c contextRecord) error {
	b, err := encode(marker(r, c))
	if err != nil {
		return err
	}
	if err := p.CompareAndSwap(contextFile, nil, b); err != nil {
		return err
	}
	keys, err := securekeys.OpenSelected(r.Backend, p, c.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = keys.Close() }()
	_, err = keylifecycle.Initialize(ctx, p, keys, c.Owner)
	return err
}
func checkContext(p statehome.Path, r record, c contextRecord) error {
	b, err := p.ReadFileLimit(contextFile, 4096)
	if err != nil {
		return err
	}
	want, err := encode(marker(r, c))
	if err != nil {
		return err
	}
	if !bytes.Equal(b, want) {
		return fmt.Errorf("bootstrap context owner mismatch")
	}
	return nil
}
func (x *runner) read() error {
	b, err := x.store.ReadFileLimit(stateFile, maxBytes)
	if err != nil {
		return err
	}
	var r record
	if err := exact(b, &r); err != nil {
		return err
	}
	if err := r.validate(x.path.Dir()); err != nil {
		return err
	}
	x.raw, x.r = b, r
	return nil
}
func (x *runner) save() error {
	if err := x.r.validate(x.path.Dir()); err != nil {
		return err
	}
	b, err := encode(x.r)
	if err != nil {
		return err
	}
	if err := x.store.CompareAndSwap(stateFile, x.raw, b); err != nil {
		return err
	}
	x.raw = b
	return nil
}

func (x *runner) context(ctx context.Context, n int) (statehome.Path, error) {
	c := x.r.Contexts[n]
	p, err := selectedPath(c)
	if err != nil {
		return p, err
	}
	if n > 0 && !c.Published {
		final, err := resolved(c.Final, statehome.Principal)
		if err != nil {
			return p, err
		}
		if _, err := os.Lstat(c.Final); err == nil {
			// A previous promotion may have committed before its sync/outcome.
			if _, err := os.Lstat(c.Staging); !errors.Is(err, os.ErrNotExist) {
				return p, fmt.Errorf("both final and staging contexts exist")
			}
			if err := x.verifyPrincipal(final, n); err != nil {
				return p, err
			}
			x.r.Contexts[n].Published = true
			if err := x.save(); err != nil {
				return p, err
			}
			return final, nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return p, err
		}
	}
	if err := p.Validate(); err != nil {
		if !errors.Is(err, os.ErrNotExist) || n == 0 || c.Published {
			return p, err
		}
		for _, k := range c.Keys {
			if k.Started {
				return p, fmt.Errorf("staging with key intent is missing; explicit recovery required")
			}
		}
		if err := p.Publish(func(stage statehome.Path) error { return initializeContext(ctx, stage, x.r, c) }); err != nil {
			return p, err
		}
	}
	return p, checkContext(p, x.r, c)
}

func (x *runner) keys(ctx context.Context, n int, p statehome.Path) error {
	c := &x.r.Contexts[n]
	s, err := securekeys.OpenSelected(x.r.Backend, p, c.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = s.Close() }()
	m, err := keylifecycle.Open(p, s, c.Owner)
	if err != nil {
		return err
	}
	for k := range c.Keys {
		entry := &c.Keys[k]
		fresh := !entry.Started
		if fresh {
			entry.Started = true
			if err := x.save(); err != nil {
				return err
			}
		}
		var result keylifecycle.Record
		if fresh {
			result, err = m.Create(ctx, entry.Slot)
		} else {
			result, err = m.Recover(ctx, entry.Slot)
		}
		if err != nil {
			return fmt.Errorf("recover %s/%s without replacing its key: %w", c.Owner, entry.Slot, err)
		}
		if result.Stage != "active" || result.Active == nil {
			return fmt.Errorf("bootstrap key is not active")
		}
		if entry.Handle != nil {
			b, _ := encode(entry.Handle)
			want, _ := encode(result.Active)
			if !bytes.Equal(b, want) {
				return fmt.Errorf("bootstrap key identity changed")
			}
		} else {
			entry.Handle = result.Active
			if err := x.save(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (x *runner) caSigner(k int, fn func(crypto.Signer) error) error {
	c := x.r.Contexts[0]
	p, err := selectedPath(c)
	if err != nil {
		return err
	}
	if err := checkContext(p, x.r, c); err != nil {
		return err
	}
	s, err := securekeys.OpenSelected(x.r.Backend, p, c.Scope)
	if err != nil {
		return err
	}
	defer func() { _ = s.Close() }()
	if c.Keys[k].Handle == nil {
		return fmt.Errorf("missing bootstrap handle")
	}
	key, err := s.Signer(*c.Keys[k].Handle)
	if err != nil {
		return err
	}
	return fn(key)
}
func (x *runner) anchor() (trust.Anchor, error) {
	k := x.r.Contexts[0].Keys[0]
	if k.Handle == nil {
		return trust.Anchor{}, fmt.Errorf("missing root handle")
	}
	fp, err := trust.RootFingerprint(ed25519.PublicKey(k.Handle.PublicKey))
	if err != nil {
		return trust.Anchor{}, err
	}
	return trust.PinRoot(k.Certificate, fp, x.now)
}

func (x *runner) certificates() error {
	for n := range x.r.Contexts {
		for k := range x.r.Contexts[n].Keys {
			entry := &x.r.Contexts[n].Keys[k]
			if len(entry.Certificate) == 0 {
				var der []byte
				if n == 0 && k == 0 {
					if err := x.caSigner(0, func(key crypto.Signer) error {
						var err error
						der, err = certificates.CreateRoot(x.r.Cluster, key, x.now)
						return err
					}); err != nil {
						return err
					}
				} else {
					a, err := x.anchor()
					if err != nil {
						return err
					}
					parent := 0
					profile := certificates.Principal
					role := trust.Role(x.r.Contexts[n].Owner)
					principal := x.r.Contexts[n].Principal
					if n == 0 {
						profile = []certificates.Profile{certificates.Root, certificates.Issuer, certificates.SnapshotSigner}[k]
						role = ""
						principal = ""
					} else {
						parent = 1
					}
					if err := x.caSigner(parent, func(key crypto.Signer) error {
						var err error
						der, err = certificates.Issue(a, x.r.Contexts[0].Keys[parent].Certificate, key, profile, role, principal, ed25519.PublicKey(entry.Handle.PublicKey), x.now)
						return err
					}); err != nil {
						return err
					}
				}
				entry.Certificate = der
				if err := x.save(); err != nil {
					return err
				}
			}
			c, id, err := certificates.Parse(entry.Certificate)
			if err != nil {
				return err
			}
			if !bytes.Equal(c.PublicKey.(ed25519.PublicKey), entry.Handle.PublicKey) || id.Cluster != x.r.Cluster {
				return fmt.Errorf("bootstrap certificate/key mismatch")
			}
			a, err := x.anchor()
			if err != nil {
				return err
			}
			if n == 0 && k == 0 {
				continue
			}
			profile := certificates.Principal
			var issuer []byte
			if n == 0 {
				profile = []certificates.Profile{certificates.Root, certificates.Issuer, certificates.SnapshotSigner}[k]
			} else {
				issuer = x.r.Contexts[0].Keys[1].Certificate
				if id.Role != trust.Role(x.r.Contexts[n].Owner) || id.Principal != x.r.Contexts[n].Principal {
					return fmt.Errorf("bootstrap principal mismatch")
				}
			}
			if _, _, err := certificates.Verify(a, entry.Certificate, issuer, profile, x.now); err != nil {
				return err
			}
		}
	}
	return nil
}

func (x *runner) report() (Report, error) {
	a, err := x.anchor()
	if err != nil {
		return Report{}, err
	}
	presentation, err := identity.Cluster(a.Fingerprint())
	if err != nil {
		return Report{}, err
	}
	r := Report{Cluster: x.r.Cluster, Identity: presentation, AuthorityDir: x.path.Dir()}
	for _, c := range x.r.Contexts[1:] {
		r.Principals = append(r.Principals, PrincipalLocation{trust.Role(c.Owner), c.Principal, c.Final})
	}
	return r, nil
}
