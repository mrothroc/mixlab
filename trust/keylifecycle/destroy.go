package keylifecycle

import (
	"context"

	"github.com/mrothroc/mixlab/statehome"
)

// Destroy is terminal cleanup for an exact active non-CA handle. The owning
// context must finish revocation/publication obligations before calling it.
// A durable destroying intent precedes deletion; Recover completes an uncertain
// deletion. Destroyed slots cannot be silently recreated by Create or Rotate.
func (m *Manager) Destroy(ctx context.Context, slot Slot, activeID string) (Record, error) {
	if slot == Root || slot == Issuer || slot == SnapshotSigner {
		return Record{}, ErrRekeyRequired
	}
	var result Record
	err := m.journal.WithProcessLock(ctx, lockName, func() error {
		old, r, err := m.read(slot)
		if err != nil {
			return err
		}
		if r.Stage != "active" || r.Active == nil || r.Active.ID != activeID {
			return statehome.ErrConflict
		}
		r.Retiring = r.Active
		r.Active = nil
		r.Stage = "destroying"
		if err := m.write(old, r); err != nil {
			return err
		}
		result, err = m.recoverLocked(slot)
		return err
	})
	return result, err
}
