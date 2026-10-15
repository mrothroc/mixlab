// Package local implements coordinator-owned durable adapters. Admission state
// and numerical execution are deliberately absent.
package local

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/statehome"
)

const maxRecord = 8 << 20

// Each transition is an immutable checksummed record. The sole mutable head
// publishes state and its receipt/round/evaluation together via one durable CAS.
type Repository struct {
	path  statehome.Path
	fault func(string) error // tests inject failures around the publication boundary
}
type head struct {
	Revision uint64 `json:"revision"`
	Hash     string `json:"hash"`
}
type record struct {
	Format   string             `json:"format"`
	Previous head               `json:"previous"`
	Change   coordinator.Change `json:"change"`
}

func Open(path statehome.Path) (*Repository, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	if _, err := path.CleanupTemporaryFiles(); err != nil {
		return nil, err
	}
	return &Repository{path: path}, nil
}
func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func key(id string) (string, error) {
	if id == "" || len(id) > 256 {
		return "", fmt.Errorf("invalid run ID")
	}
	return digest([]byte(id)), nil
}
func decode(b []byte, dst any) error {
	if len(b) > maxRecord {
		return fmt.Errorf("coordinator record exceeds bound")
	}
	if err := json.Unmarshal(b, dst); err != nil {
		return err
	}
	want, err := json.Marshal(dst)
	if err != nil {
		return err
	}
	if !bytes.Equal(b, want) {
		return fmt.Errorf("noncanonical coordinator record")
	}
	return nil
}
func (r *Repository) current(id string) (head, []byte, error) {
	k, err := key(id)
	if err != nil {
		return head{}, nil, err
	}
	b, err := r.path.ReadFileLimit(k+".head", 1024)
	if errors.Is(err, os.ErrNotExist) {
		return head{}, nil, coordinator.ErrNotFound
	}
	if err != nil {
		return head{}, nil, err
	}
	var h head
	if err = decode(b, &h); err != nil {
		return h, nil, err
	}
	if h.Revision == 0 || len(h.Hash) != 64 {
		return h, nil, fmt.Errorf("invalid coordinator head")
	}
	return h, b, nil
}
func (r *Repository) read(id string, h head) (record, error) {
	var v record
	k, err := key(id)
	if err != nil {
		return v, err
	}
	if b, err := hex.DecodeString(h.Hash); err != nil || len(b) != 32 || hex.EncodeToString(b) != h.Hash {
		return v, fmt.Errorf("invalid record digest")
	}
	b, err := r.path.ReadFileLimit(k+"-"+h.Hash+".record", maxRecord)
	if err != nil {
		return v, err
	}
	if digest(b) != h.Hash {
		return v, fmt.Errorf("coordinator record checksum mismatch")
	}
	if err = decode(b, &v); err != nil {
		return v, err
	}
	if v.Format != "mixlab_coordinator_change_v1" || v.Change.Next.Revision != h.Revision || v.Previous.Revision != h.Revision-1 || v.Change.Next.Plan.Cohort.RunID != id || (h.Revision == 1 && v.Previous.Hash != "") || (h.Revision > 1 && len(v.Previous.Hash) != 64) {
		return v, fmt.Errorf("invalid coordinator chain")
	}
	return v, validateChange(v.Change)
}
func (r *Repository) Load(ctx context.Context, id string) (coordinator.State, error) {
	if err := ctx.Err(); err != nil {
		return coordinator.State{}, err
	}
	h, _, err := r.current(id)
	if err != nil {
		return coordinator.State{}, err
	}
	v, err := r.read(id, h)
	return v.Change.Next, err
}
func (r *Repository) Receipt(ctx context.Context, id, update string) (learner.Receipt, error) {
	h, _, err := r.current(id)
	if err != nil {
		return learner.Receipt{}, err
	}
	for h.Revision > 0 {
		if err := ctx.Err(); err != nil {
			return learner.Receipt{}, err
		}
		v, err := r.read(id, h)
		if err != nil {
			return learner.Receipt{}, err
		}
		if v.Change.Receipt != nil && v.Change.Receipt.UpdateID == update {
			return *v.Change.Receipt, nil
		}
		h = v.Previous
	}
	return learner.Receipt{}, coordinator.ErrNotFound
}
func (r *Repository) Save(ctx context.Context, id string, revision uint64, c coordinator.Change) error {
	k, err := key(id)
	if err != nil {
		return err
	}
	return r.path.WithProcessLock(ctx, k+".lock", func() error {
		h, old, err := r.current(id)
		if errors.Is(err, coordinator.ErrNotFound) {
			h = head{}
			old = nil
		} else if err != nil {
			return err
		}
		if revision == ^uint64(0) || h.Revision != revision || c.Next.Revision != revision+1 || c.Next.Plan.Cohort.RunID != id {
			return coordinator.ErrConflict
		}
		if err := validateChange(c); err != nil {
			return err
		}
		if revision > 0 {
			prev, err := r.read(id, h)
			if err != nil {
				return err
			}
			if !reflect.DeepEqual(prev.Change.Next.Plan, c.Next.Plan) || !reflect.DeepEqual(prev.Change.Next.Policy, c.Next.Policy) || (prev.Change.Next.Commit != nil && !reflect.DeepEqual(prev.Change.Next.Commit, c.Next.Commit)) {
				return fmt.Errorf("immutable run contract changed")
			}
		}
		b, err := json.Marshal(record{"mixlab_coordinator_change_v1", h, c})
		if err != nil {
			return err
		}
		if len(b) > maxRecord {
			return fmt.Errorf("coordinator change too large")
		}
		next := head{c.Next.Revision, digest(b)}
		name := k + "-" + next.Hash + ".record"
		if err = r.path.CompareAndSwap(name, nil, b); errors.Is(err, statehome.ErrConflict) {
			existing, e := r.path.ReadFileLimit(name, maxRecord)
			if e != nil {
				return e
			}
			if !bytes.Equal(existing, b) {
				return fmt.Errorf("immutable record mismatch")
			}
			err = nil
		}
		if err != nil {
			return err
		}
		if r.fault != nil {
			if err = r.fault("record"); err != nil {
				return err
			}
		}
		if err = ctx.Err(); err != nil {
			return err
		}
		encoded, _ := json.Marshal(next)
		if err = r.path.CompareAndSwap(k+".head", old, encoded); errors.Is(err, statehome.ErrConflict) {
			return coordinator.ErrConflict
		}
		if err != nil {
			return err
		}
		if r.fault != nil {
			return r.fault("head")
		}
		return nil
	})
}

func validateChange(c coordinator.Change) error {
	if err := c.Next.Validate(); err != nil {
		return err
	}
	if c.Receipt != nil {
		if err := c.Receipt.Validate(); err != nil {
			return err
		}
		found := false
		for _, u := range c.Next.Updates {
			h, _ := json.Marshal(u)
			if u.ID == c.Receipt.UpdateID && digest(h) == c.Receipt.UpdateHash && u.PrivateHash == c.Receipt.PrivateHash && u.Binding == c.Receipt.Binding {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("receipt not joined to accepted update")
		}
	}
	if c.Published != nil && (c.Next.Phase != coordinator.Evaluating || !reflect.DeepEqual(c.Next.Candidate, c.Published)) {
		return fmt.Errorf("round not joined to published state")
	}
	if (c.Evaluation == nil) != (c.Decision == nil) {
		return fmt.Errorf("evaluation and decision must publish together")
	}
	if c.Evaluation != nil {
		if err := c.Evaluation.Validate(); err != nil {
			return err
		}
		if err := c.Decision.Validate(); err != nil {
			return err
		}
		b, _ := json.Marshal(c.Evaluation)
		if c.Next.Evaluation == nil || c.Evaluation.Assignment != *c.Next.Evaluation || c.Decision.ResultHash != digest(b) || c.Decision.Binding != c.Evaluation.Assignment.Binding || c.Decision.Round != c.Evaluation.Assignment.Round || c.Next.Round != c.Decision.Round+1 || (c.Decision.Decision == "complete" && c.Next.Phase != coordinator.Complete) || (c.Decision.Decision == "continue" && c.Next.Phase != coordinator.Ready) {
			return fmt.Errorf("evaluation decision/state mismatch")
		}
	}
	return nil
}
