package local

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
)

// Register persists a staged slot's immutable registration before cohort commit.
// Authentication and original-allocation liveness remain the admission adapter's
// obligation. This does not commit a cohort or authorize any training.
func (r *Repository) Register(ctx context.Context, v run.Registration) error {
	k, err := key(v.RunID)
	if err != nil {
		return err
	}
	return r.path.WithProcessLock(ctx, k+".lock", func() error {
		s, err := r.Load(ctx, v.RunID)
		if err != nil {
			return err
		}
		if err = v.ValidateFor(s.Plan); err != nil {
			return err
		}
		b, err := json.Marshal(v)
		if err != nil {
			return err
		}
		name := k + "-" + digest([]byte(v.Slot.Member.WorkerID)) + ".registration"
		old, err := r.path.ReadFileLimit(name, maxRecord)
		if err == nil {
			if bytes.Equal(old, b) {
				return nil
			}
			return coordinator.ErrConflict
		}
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if s.Phase != coordinator.Staged {
			return fmt.Errorf("registration requires staged plan")
		}
		if err = r.path.CompareAndSwap(name, nil, b); errors.Is(err, statehome.ErrConflict) {
			old, e := r.path.ReadFileLimit(name, maxRecord)
			if e != nil {
				return e
			}
			if bytes.Equal(old, b) {
				return nil
			}
			return coordinator.ErrConflict
		}
		return err
	})
}
func (r *Repository) Registrations(ctx context.Context, id string) ([]run.Registration, error) {
	s, err := r.Load(ctx, id)
	if err != nil {
		return nil, err
	}
	k, _ := key(id)
	out := []run.Registration{}
	for _, slot := range s.Plan.Slots {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		b, err := r.path.ReadFileLimit(k+"-"+digest([]byte(slot.Member.WorkerID))+".registration", maxRecord)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		var v run.Registration
		if err = decode(b, &v); err != nil {
			return nil, err
		}
		if err = v.ValidateFor(s.Plan); err != nil {
			return nil, err
		}
		if v.Slot != slot {
			return nil, fmt.Errorf("registration slot mismatch")
		}
		out = append(out, v)
	}
	return out, nil
}

// PruneOrphans removes only unpublished transition records for one run, while
// excluding all cooperating writers. Committed history and registrations stay.
func (r *Repository) PruneOrphans(ctx context.Context, id string) (int, error) {
	k, err := key(id)
	if err != nil {
		return 0, err
	}
	removed := 0
	err = r.path.WithProcessLock(ctx, k+".lock", func() error {
		h, _, err := r.current(id)
		if errors.Is(err, coordinator.ErrNotFound) {
			h = head{}
		} else if err != nil {
			return err
		}
		live := map[string]bool{}
		for h.Revision > 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			v, err := r.read(id, h)
			if err != nil {
				return err
			}
			live[k+"-"+h.Hash+".record"] = true
			h = v.Previous
		}
		entries, err := os.ReadDir(r.path.Dir())
		if err != nil {
			return err
		}
		for _, entry := range entries {
			name := entry.Name()
			if !strings.HasPrefix(name, k+"-") || !strings.HasSuffix(name, ".record") || live[name] {
				continue
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			b, err := r.path.ReadFileLimit(name, maxRecord)
			if err != nil {
				return err
			}
			if err = r.path.CompareAndSwap(name, b, nil); err != nil {
				return err
			}
			removed++
		}
		return nil
	})
	return removed, err
}
