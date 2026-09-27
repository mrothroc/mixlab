// Package keylifecycle owns durable local key intent and retirement, not
// certificate issuance, enrollment approval, revocation, or application grants.
package keylifecycle

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/securekeys"
	"github.com/mrothroc/mixlab/statehome"
)

type Slot string

const (
	Root           Slot = "root"
	Issuer         Slot = "issuer"
	SnapshotSigner Slot = "snapshot-signer"
	Principal      Slot = "principal"
	NodeEnvelope   Slot = "node-envelope"
	Workload       Slot = "workload"
)
const version = "mixlab_key_lifecycle_v1"
const lockName = "key-lifecycle.lock"
const contextName = "key-context.json"

var ErrRecoveryRequired = errors.New("key intent needs explicit recovery or abort; never regenerate implicitly")
var ErrRekeyRequired = errors.New("CA key replacement requires explicit cluster rekey, not ordinary rotation")

type Record struct {
	Version    string             `json:"version"`
	Owner      string             `json:"owner"`
	Scope      string             `json:"scope"`
	Backend    string             `json:"backend"`
	Slot       Slot               `json:"slot"`
	Generation uint64             `json:"generation"`
	Stage      string             `json:"stage"`
	Active     *securekeys.Handle `json:"active"`
	Candidate  *securekeys.Handle `json:"candidate"`
	Retiring   *securekeys.Handle `json:"retiring"`
}

type journal interface {
	ReadFileLimit(string, int64) ([]byte, error)
	CompareAndSwap(string, []byte, []byte) error
	WithProcessLock(context.Context, string, func() error) error
}
type Manager struct {
	journal journal
	keys    *securekeys.Store
	owner   string
}

type ownerRecord struct {
	Version string `json:"version"`
	Owner   string `json:"owner"`
	Scope   string `json:"scope"`
	Backend string `json:"backend"`
}

func configured(path statehome.Path, keys *securekeys.Store, owner string) (*Manager, error) {
	if keys == nil || !validOwner(owner) {
		return nil, fmt.Errorf("invalid key lifecycle owner/store")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Manager{path, keys, owner}, nil
}

// Initialize is only for an explicitly new lifecycle context. Existing context
// metadata or orphaned intent records reject initialization, never reset them.
func Initialize(ctx context.Context, path statehome.Path, keys *securekeys.Store, owner string) (*Manager, error) {
	m, err := configured(path, keys, owner)
	if err != nil {
		return nil, err
	}
	err = path.WithProcessLock(ctx, lockName, func() error {
		if _, err := path.ReadFileLimit(contextName, 1024); err == nil {
			return statehome.ErrExists
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		for _, slot := range []Slot{Root, Issuer, SnapshotSigner, Principal, NodeEnvelope, Workload} {
			if _, err := path.ReadFileLimit(name(slot), 4096); err == nil {
				return ErrRecoveryRequired
			} else if !errors.Is(err, os.ErrNotExist) {
				return err
			}
		}
		b, err := json.Marshal(ownerRecord{version, owner, keys.Scope(), keys.Backend()})
		if err != nil {
			return err
		}
		return path.CompareAndSwap(contextName, nil, b)
	})
	if err != nil {
		return nil, err
	}
	return m, nil
}

func (m *Manager) checkContext() error {
	b, err := m.journal.ReadFileLimit(contextName, 1024)
	if err != nil {
		return fmt.Errorf("%w: key context unavailable (%v)", ErrRecoveryRequired, err)
	}
	want, err := json.Marshal(ownerRecord{version, m.owner, m.keys.Scope(), m.keys.Backend()})
	if err != nil {
		return err
	}
	if !bytes.Equal(b, want) {
		return fmt.Errorf("key lifecycle owner/backend/scope mismatch")
	}
	return nil
}

// Open requires the explicit backend for this persisted context. No identity,
// key, or directory is created here. Owner describes the local bounded context.
func Open(path statehome.Path, keys *securekeys.Store, owner string) (*Manager, error) {
	m, err := configured(path, keys, owner)
	if err != nil {
		return nil, err
	}
	if err := m.checkContext(); err != nil {
		return nil, err
	}
	return m, nil
}
func validOwner(owner string) bool {
	switch owner {
	case "cluster-authority", "authority", "controller", "coordinator", "node", "worker":
		return true
	}
	return false
}
func (m *Manager) allowed(slot Slot) bool {
	switch slot {
	case Root, Issuer, SnapshotSigner:
		return m.owner == "cluster-authority"
	case Principal:
		return m.owner == "authority" || m.owner == "controller" || m.owner == "coordinator" || m.owner == "node"
	case NodeEnvelope:
		return m.owner == "node"
	case Workload:
		return m.owner == "worker"
	}
	return false
}
func name(slot Slot) string { return "key-intent-" + string(slot) + ".json" }

func (m *Manager) validate(r Record, slot Slot) error {
	if !m.allowed(slot) || r.Version != version || r.Owner != m.owner || r.Scope != m.keys.Scope() || r.Backend != m.keys.Backend() || r.Slot != slot || r.Generation == 0 {
		return fmt.Errorf("key intent context mismatch")
	}
	profile := securekeys.Version
	if slot == NodeEnvelope {
		profile = securekeys.EnvelopeVersion
	}
	seen := map[string]bool{}
	for _, h := range []*securekeys.Handle{r.Active, r.Candidate, r.Retiring} {
		if h == nil {
			continue
		}
		id, err := hex.DecodeString(h.ID)
		if err != nil || len(id) != 16 || hex.EncodeToString(id) != h.ID || h.Version != profile || h.Backend != r.Backend || h.Scope != r.Scope || len(h.PublicKey) != 32 || seen[h.ID] {
			return fmt.Errorf("invalid key intent handle")
		}
		seen[h.ID] = true
	}
	ok := false
	switch r.Stage {
	case "creating":
		ok = r.Active == nil && r.Candidate != nil && r.Retiring == nil
	case "active":
		ok = r.Active != nil && r.Candidate == nil && r.Retiring == nil
	case "rotating", "rotation-ready":
		ok = r.Active != nil && r.Candidate != nil && r.Retiring == nil
	case "retiring", "deleting-retired":
		ok = r.Active != nil && r.Candidate == nil && r.Retiring != nil
	case "aborting":
		ok = r.Candidate != nil && r.Retiring == nil
	case "aborted":
		ok = r.Active == nil && r.Candidate == nil && r.Retiring == nil
	case "destroying":
		ok = r.Active == nil && r.Candidate == nil && r.Retiring != nil && slot != Root && slot != Issuer && slot != SnapshotSigner
	case "destroyed":
		ok = r.Active == nil && r.Candidate == nil && r.Retiring == nil && slot != Root && slot != Issuer && slot != SnapshotSigner
	}
	if !ok {
		return fmt.Errorf("invalid key lifecycle stage/handles")
	}
	return nil
}
func (m *Manager) read(slot Slot) ([]byte, Record, error) {
	var r Record
	if err := m.checkContext(); err != nil {
		return nil, r, err
	}
	if !m.allowed(slot) {
		return nil, r, fmt.Errorf("key slot not owned by context")
	}
	b, err := m.journal.ReadFileLimit(name(slot), 4096)
	if err != nil {
		return nil, r, err
	}
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	canonical, err := json.Marshal(r)
	if err != nil || !bytes.Equal(canonical, b) {
		return nil, r, fmt.Errorf("invalid key intent encoding")
	}
	return b, r, m.validate(r, slot)
}
func (m *Manager) write(old []byte, r Record) error {
	if err := m.validate(r, r.Slot); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	return m.journal.CompareAndSwap(name(r.Slot), old, b)
}

// View exposes public recovery status, not private keys or an authorization grant.
func (m *Manager) View(slot Slot) (Record, error) { _, r, err := m.read(slot); return r, err }

func (m *Manager) Create(ctx context.Context, slot Slot) (Record, error) {
	return m.begin(ctx, slot, false)
}
func (m *Manager) Rotate(ctx context.Context, slot Slot) (Record, error) {
	if slot == Root || slot == Issuer || slot == SnapshotSigner {
		return Record{}, ErrRekeyRequired
	}
	return m.begin(ctx, slot, true)
}
func (m *Manager) begin(ctx context.Context, slot Slot, rotate bool) (Record, error) {
	var result Record
	err := m.journal.WithProcessLock(ctx, lockName, func() error {
		old, r, err := m.read(slot)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if rotate {
			if err != nil || r.Stage != "active" {
				return ErrRecoveryRequired
			}
			if err := m.keys.Inspect(*r.Active); err != nil {
				return err
			}
		} else if err == nil && r.Stage != "aborted" {
			return statehome.ErrExists
		}
		generation := r.Generation + 1
		if generation == 0 {
			return fmt.Errorf("key generation exhausted")
		}
		var pending *securekeys.Pending
		if slot == NodeEnvelope {
			pending, err = m.keys.PrepareEnvelope()
		} else {
			pending, err = m.keys.PrepareSigning()
		}
		if err != nil {
			return err
		}
		defer pending.Close()
		h := pending.Handle()
		r = Record{Version: version, Owner: m.owner, Scope: m.keys.Scope(), Backend: m.keys.Backend(), Slot: slot, Generation: generation, Stage: "creating", Active: r.Active, Candidate: &h}
		if rotate {
			r.Stage = "rotating"
		}
		if err := m.write(old, r); err != nil {
			return err
		}
		// Publication must never precede the durable candidate handle.
		if err := pending.Publish(); err != nil {
			return err
		}
		result, err = m.recoverLocked(slot)
		return err
	})
	return result, err
}

func (m *Manager) Recover(ctx context.Context, slot Slot) (Record, error) {
	var result Record
	err := m.journal.WithProcessLock(ctx, lockName, func() error { r, err := m.recoverLocked(slot); result = r; return err })
	return result, err
}
func (m *Manager) recoverLocked(slot Slot) (Record, error) {
	old, r, err := m.read(slot)
	if err != nil {
		return r, err
	}
	if r.Active != nil {
		if err := m.keys.Inspect(*r.Active); err != nil {
			return r, err
		}
	}
	switch r.Stage {
	case "creating", "rotating":
		if err := m.keys.Inspect(*r.Candidate); err != nil {
			return r, errors.Join(ErrRecoveryRequired, err)
		}
		if r.Stage == "creating" {
			r.Active = r.Candidate
			r.Candidate = nil
			r.Stage = "active"
		} else {
			r.Stage = "rotation-ready"
		}
	case "rotation-ready":
		return r, m.keys.Inspect(*r.Candidate)
	case "retiring":
		return r, m.keys.Inspect(*r.Retiring)
	case "aborting", "deleting-retired", "destroying":
		destroying := r.Stage == "destroying"
		h := r.Candidate
		if r.Stage == "deleting-retired" || destroying {
			h = r.Retiring
		}
		if err := m.keys.Inspect(*h); err == nil {
			if err = m.keys.Delete(*h); err != nil {
				return r, err
			}
		} else if !errors.Is(err, securekeys.ErrMissing) {
			return r, err
		}
		r.Candidate = nil
		r.Retiring = nil
		r.Stage = "active"
		if r.Active == nil {
			r.Stage = "aborted"
		}
		if destroying {
			r.Stage = "destroyed"
		}
	default:
		return r, nil
	}
	if err := m.write(old, r); err != nil {
		return r, err
	}
	return r, nil
}

// Activate is called only after the owning enrollment workflow has durably
// published the new credential. It does not destroy the previous key; Retire
// follows after that owner has reconciled certificate/revocation obligations.
func (m *Manager) Activate(ctx context.Context, slot Slot, candidateID string) (Record, error) {
	return m.transition(ctx, slot, "activate", candidateID)
}
func (m *Manager) Abort(ctx context.Context, slot Slot, candidateID string) (Record, error) {
	return m.transition(ctx, slot, "abort", candidateID)
}
func (m *Manager) Retire(ctx context.Context, slot Slot, retiringID string) (Record, error) {
	return m.transition(ctx, slot, "retire", retiringID)
}
func (m *Manager) transition(ctx context.Context, slot Slot, action, id string) (Record, error) {
	var result Record
	err := m.journal.WithProcessLock(ctx, lockName, func() error {
		old, r, err := m.read(slot)
		if err != nil {
			return err
		}
		if r.Active != nil {
			if err := m.keys.Inspect(*r.Active); err != nil {
				return err
			}
		}
		switch action {
		case "activate":
			if r.Stage != "rotation-ready" || r.Candidate.ID != id {
				return statehome.ErrConflict
			}
			if err := m.keys.Inspect(*r.Candidate); err != nil {
				return err
			}
			r.Retiring = r.Active
			r.Active = r.Candidate
			r.Candidate = nil
			r.Stage = "retiring"
		case "abort":
			if (r.Stage != "creating" && r.Stage != "rotating" && r.Stage != "rotation-ready") || r.Candidate.ID != id {
				return statehome.ErrConflict
			}
			r.Stage = "aborting"
		case "retire":
			if r.Stage != "retiring" || r.Retiring.ID != id {
				return statehome.ErrConflict
			}
			r.Stage = "deleting-retired"
		}
		if err := m.write(old, r); err != nil {
			return err
		}
		result, err = m.recoverLocked(slot)
		return err
	})
	return result, err
}
