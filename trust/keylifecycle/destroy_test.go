package keylifecycle

import (
	"errors"
	"testing"

	"github.com/mrothroc/mixlab/securekeys"
)

func TestTerminalKeyDestruction(t *testing.T) {
	for _, slot := range []Slot{Principal, NodeEnvelope} {
		t.Run(string(slot), func(t *testing.T) {
			m, p, k := setup(t, "node")
			r, err := m.Create(ctx, slot)
			require(t, err)
			h := *r.Active
			if _, err := m.Destroy(ctx, slot, "wrong"); err == nil {
				t.Fatal("destroyed another key")
			}
			require(t, k.Inspect(h))
			r, err = m.Destroy(ctx, slot, h.ID)
			require(t, err)
			if r.Stage != "destroyed" || r.Active != nil {
				t.Fatal(r)
			}
			if err := k.Inspect(h); !errors.Is(err, securekeys.ErrMissing) {
				t.Fatal("key survives destruction", err)
			}
			m, err = Open(p, k, "node")
			require(t, err)
			r, err = m.Recover(ctx, slot)
			require(t, err)
			if r.Stage != "destroyed" {
				t.Fatal("destruction not durable")
			}
			if _, err := m.Create(ctx, slot); err == nil {
				t.Fatal("destroyed key silently regenerated")
			}
		})
	}
	for _, slot := range []Slot{Root, Issuer, SnapshotSigner} {
		m, _, _ := setup(t, "cluster-authority")
		if _, err := m.Destroy(ctx, slot, "any"); !errors.Is(err, ErrRekeyRequired) {
			t.Fatal("CA destruction bypassed rekey", err)
		}
	}
}

func TestDestroyRecoveryAfterPrivateDeletion(t *testing.T) {
	m, _, k := setup(t, "node")
	r, err := m.Create(ctx, Principal)
	require(t, err)
	h := *r.Active
	old, r, err := m.read(Principal)
	require(t, err)
	r.Retiring = r.Active
	r.Active = nil
	r.Stage = "destroying"
	require(t, m.write(old, r))
	require(t, k.Delete(h))
	r, err = m.Recover(ctx, Principal)
	require(t, err)
	if r.Stage != "destroyed" {
		t.Fatal("uncertain deletion not reconciled")
	}
}
