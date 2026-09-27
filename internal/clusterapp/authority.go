// Package clusterapp composes context-owned trust services for the cluster CLI.
// It contains no training runtime, discovery policy, or node scheduling rules.
package clusterapp

import (
	"bytes"
	"context"
	"crypto"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/transport/enrollmenttls"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/authority"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/trust/workload"
)

const runtimeFile = "authority-runtime.json"
const runtimeLock = "authority-runtime.lock"
const runtimeVersion = "mixlab_authority_runtime_v1"

type runtimeState struct {
	Version     string `json:"version"`
	Fingerprint string `json:"fingerprint"`
	Ready       bool   `json:"ready"`
	Workloads   bool   `json:"workloads,omitempty"`
}

type Authority struct {
	currentMu     sync.Mutex
	Anchor        trust.Anchor
	Snapshots     *authority.Store
	Enrollment    *enrollment.Service
	Workloads     *workload.Service
	Principal     *principal.Store
	keys          *securekeys.Store
	principalPath statehome.Path
}

func (a *Authority) Close() error { return errors.Join(a.Principal.Close(), a.keys.Close()) }

func runtimeMarker(p statehome.Path, pin string) ([]byte, runtimeState, error) {
	b, err := p.ReadFileLimit(runtimeFile, 4096)
	if err != nil {
		return nil, runtimeState{}, err
	}
	var r runtimeState
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, fmt.Errorf("invalid authority runtime marker")
	}
	want, _ := json.Marshal(r)
	if !bytes.Equal(b, want) || r.Version != runtimeVersion || r.Fingerprint != pin {
		return nil, r, fmt.Errorf("authority runtime marker mismatch")
	}
	return b, r, nil
}

// InitializeAuthority is invoked only by explicit init/recover. A durable ready
// marker prohibits recreating a deleted revocation or enrollment journal.
func InitializeAuthority(ctx context.Context, p statehome.Path, now time.Time) error {
	m, err := bootstrap.LoadAuthority(p, now)
	if err != nil {
		return err
	}
	return p.WithProcessLock(ctx, runtimeLock, func() error {
		old, r, err := runtimeMarker(p, m.Anchor.Fingerprint())
		if errors.Is(err, os.ErrNotExist) {
			r = runtimeState{Version: runtimeVersion, Fingerprint: m.Anchor.Fingerprint()}
			old, err = json.Marshal(r)
			if err != nil {
				return err
			}
			if err = p.CompareAndSwap(runtimeFile, nil, old); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
		a, err := composeAuthority(ctx, p, m, !r.Ready, !r.Workloads, now)
		if err != nil {
			return err
		}
		defer func() { _ = a.Close() }()
		if r.Ready && r.Workloads {
			return nil
		}
		r.Ready = true
		r.Workloads = true
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		return p.CompareAndSwap(runtimeFile, old, b)
	})
}

func OpenAuthority(ctx context.Context, p statehome.Path, now time.Time) (*Authority, error) {
	m, err := bootstrap.LoadAuthority(p, now)
	if err != nil {
		return nil, err
	}
	_, r, err := runtimeMarker(p, m.Anchor.Fingerprint())
	if err != nil {
		return nil, err
	}
	if !r.Ready || !r.Workloads {
		return nil, fmt.Errorf("authority initialization incomplete; run explicit init recovery")
	}
	return composeAuthority(ctx, p, m, false, false, now)
}

func activeCAKey(manager *keylifecycle.Manager, slot keylifecycle.Slot, h securekeys.Handle) error {
	r, err := manager.View(slot)
	if err != nil {
		return err
	}
	b, _ := json.Marshal(r.Active)
	want, _ := json.Marshal(h)
	if r.Stage != "active" || !bytes.Equal(b, want) {
		return fmt.Errorf("authority key lifecycle differs from bootstrap")
	}
	return nil
}

func composeAuthority(ctx context.Context, p statehome.Path, m bootstrap.AuthorityMaterial, initialize, initializeWorkloads bool, now time.Time) (*Authority, error) {
	keys, err := securekeys.OpenSelected(m.IssuerKey.Backend, p, m.IssuerKey.Scope)
	if err != nil {
		return nil, err
	}
	keep := false
	defer func() {
		if !keep {
			_ = keys.Close()
		}
	}()
	manager, err := keylifecycle.Open(p, keys, "cluster-authority")
	if err != nil {
		return nil, err
	}
	if err := activeCAKey(manager, keylifecycle.Issuer, m.IssuerKey); err != nil {
		return nil, err
	}
	if err := activeCAKey(manager, keylifecycle.SnapshotSigner, m.SnapshotKey); err != nil {
		return nil, err
	}
	issuer, err := keys.Signer(m.IssuerKey)
	if err != nil {
		return nil, err
	}
	snapshotKey, err := keys.Signer(m.SnapshotKey)
	if err != nil {
		return nil, err
	}
	snapshots, err := authority.Open(p, m.Anchor, m.SnapshotSigner, snapshotKey, now)
	if initialize && errors.Is(err, os.ErrNotExist) {
		snapshots, err = authority.Initialize(ctx, p, m.Anchor, m.SnapshotSigner, snapshotKey, m.InitialSnapshot, now)
	}
	if err != nil {
		return nil, err
	}
	current, err := snapshots.Refresh(ctx, now)
	if err != nil {
		return nil, err
	}
	identity, err := principal.Open(m.AuthorityPrincipal, now)
	if err != nil {
		return nil, err
	}
	defer func() {
		if !keep {
			_ = identity.Close()
		}
	}()
	if err := refreshPrincipal(ctx, identity, current, now); err != nil {
		return nil, err
	}
	state, signer, err := identity.Active(now)
	if err != nil {
		return nil, err
	}
	v, err := trust.VerifySnapshot(m.Anchor, current, now)
	if err != nil {
		return nil, err
	}
	opts := enrollment.Authority{Anchor: m.Anchor, PrincipalChain: state.Chain, PrincipalKey: signer, Issuer: m.Issuer, IssuerKey: issuer,
		CurrentPrincipal: func(now time.Time) ([][]byte, crypto.Signer, error) {
			s, k, err := identity.Active(now)
			return s.Chain, k, err
		},
	}
	enrollments, err := enrollment.Open(p, opts, v, now)
	if initialize && errors.Is(err, os.ErrNotExist) {
		enrollments, err = enrollment.Initialize(ctx, p, opts, v, now)
	}
	if err != nil {
		return nil, err
	}
	workloadOptions := workload.Authority{Anchor: m.Anchor, Issuer: m.Issuer, Key: issuer}
	workloads, err := workload.Open(p, workloadOptions, v, now)
	if initializeWorkloads && errors.Is(err, os.ErrNotExist) {
		workloads, err = workload.Initialize(ctx, p, workloadOptions, v, now)
	}
	if err != nil {
		return nil, err
	}
	keep = true
	return &Authority{Anchor: m.Anchor, Snapshots: snapshots, Enrollment: enrollments, Workloads: workloads, Principal: identity, keys: keys, principalPath: m.AuthorityPrincipal}, nil
}

func refreshPrincipal(ctx context.Context, p *principal.Store, s trust.SignedSnapshot, now time.Time) error {
	return p.Refresh(ctx, s, now)
}

func (a *Authority) Current(ctx context.Context, now time.Time) (trust.VerifiedSnapshot, error) {
	a.currentMu.Lock()
	defer a.currentMu.Unlock()
	s, err := a.Snapshots.Refresh(ctx, now)
	if err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	if err := refreshPrincipal(ctx, a.Principal, s, now); err != nil {
		return trust.VerifiedSnapshot{}, err
	}
	return trust.VerifySnapshot(a.Anchor, s, now)
}

// EnrollmentSource refreshes a separately owned coordinator identity from the
// local authority without copying any CA handles into the coordinator context.
func (a *Authority) EnrollmentSource(ctx context.Context, p *principal.Store, now time.Time) (enrollmenttls.Identity, error) {
	if _, err := a.Current(ctx, now); err != nil {
		return enrollmenttls.Identity{}, err
	}
	snapshot, err := a.Snapshots.Load(now)
	if err != nil {
		return enrollmenttls.Identity{}, err
	}
	if err := refreshPrincipal(ctx, p, snapshot, now); err != nil {
		return enrollmenttls.Identity{}, err
	}
	s, k, err := p.Active(now)
	if err != nil {
		return enrollmenttls.Identity{}, err
	}
	if s.Role != trust.Coordinator || s.Fingerprint != a.Anchor.Fingerprint() {
		return enrollmenttls.Identity{}, fmt.Errorf("matching coordinator identity required")
	}
	return enrollmenttls.Identity{Chain: s.Chain, Key: k}, nil
}
