package local

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/statehome"
)

// Checkpoints is a replayable catalog of published global model/outer-state
// references. It does not contain learner-private optimizer or sampler state.
type Checkpoints struct{ path statehome.Path }

func OpenCheckpoints(path statehome.Path) (*Checkpoints, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Checkpoints{path}, nil
}

// Sync repairs the catalog from the authoritative publication chain. The
// cursor advances only after every published reference is durable, including
// rounds completed between service ticks or just before a service crash.
func (c *Checkpoints) Sync(ctx context.Context, r *Repository, id string) error {
	k, err := key(id)
	if err != nil {
		return err
	}
	return c.path.WithProcessLock(ctx, k+".catalog.lock", func() error {
		current, _, err := r.current(id)
		if err != nil {
			return err
		}
		name := k + ".catalog.head"
		old, err := c.path.ReadFileLimit(name, 1024)
		var previous head
		if errors.Is(err, os.ErrNotExist) {
			old = nil
		} else if err != nil {
			return err
		} else if err = decode(old, &previous); err != nil {
			return err
		}
		if previous.Revision > current.Revision {
			return fmt.Errorf("checkpoint catalog is ahead of coordinator history")
		}
		h := current
		for h.Revision > previous.Revision {
			if err := ctx.Err(); err != nil {
				return err
			}
			v, err := r.read(id, h)
			if err != nil {
				return err
			}
			if v.Change.Published != nil {
				if err = c.Capture(ctx, v.Change.Next); err != nil {
					return err
				}
			}
			h = v.Previous
		}
		if h != previous {
			return fmt.Errorf("checkpoint catalog differs from coordinator history")
		}
		if current == previous {
			return nil
		}
		b, err := json.Marshal(current)
		if err != nil {
			return err
		}
		return c.path.CompareAndSwap(name, old, b)
	})
}
func (c *Checkpoints) Capture(ctx context.Context, s coordinator.State) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := s.Validate(); err != nil {
		return err
	}
	if s.Candidate == nil || s.Phase == coordinator.Aggregating || s.Phase == coordinator.Publishing {
		return nil
	}
	if err := s.Candidate.Validate(); err != nil {
		return err
	}
	key, err := key(s.Plan.Cohort.RunID)
	if err != nil {
		return err
	}
	name := fmt.Sprintf("%s-%020d.global.json", key, s.Candidate.Round)
	b, err := json.Marshal(s.Candidate)
	if err != nil {
		return err
	}
	err = c.path.CompareAndSwap(name, nil, b)
	if errors.Is(err, statehome.ErrConflict) {
		old, e := c.path.ReadFileLimit(name, maxRecord)
		if e != nil {
			return e
		}
		if !bytes.Equal(old, b) {
			return fmt.Errorf("global checkpoint identity changed")
		}
		return nil
	}
	return err
}
