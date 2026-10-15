// Package artifactapi owns artifact authorization and grant-use accounting, not
// byte storage or numerical execution. Every request rechecks current trust.
package artifactapi

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

type Ledger struct{ path statehome.Path }
type reservation struct {
	Format string              `json:"format"`
	Grant  run.Grant           `json:"grant"`
	Proof  trust.AcceptedProof `json:"proof"`
	Uses   []run.GrantUse      `json:"uses"`
}

func OpenLedger(path statehome.Path) (*Ledger, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	if _, err := path.CleanupTemporaryFiles(); err != nil {
		return nil, err
	}
	return &Ledger{path}, nil
}
func hash(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

// WithUse durably consumes a use before invoking I/O. Failed/interrupted I/O
// keeps its reservation; exact retries reuse it, never refund it. Each use is
// serialized across processes while different uses may transfer concurrently.
func (l *Ledger) WithUse(ctx context.Context, g run.Grant, u run.GrantUse, proof trust.AcceptedProof, fn func() error) error {
	if err := g.Validate(); err != nil {
		return err
	}
	if err := u.Validate(); err != nil {
		return err
	}
	if fn == nil || proof.Digest == "" || proof.Generation == 0 || u.GrantID != g.ID || u.GrantHash != g.SHA256 || u.Operation != g.Operation || u.UploadID != g.UploadID {
		return fmt.Errorf("grant-use binding mismatch")
	}
	key := hash(g.Coordinator.ClusterID + "\x00" + g.Coordinator.PrincipalID + "\x00" + g.ID)
	// Reserve before creating a per-use lock inode: rejected random use IDs must
	// not consume unbounded disk space after a grant's budget is exhausted.
	if err := l.reserve(ctx, key, g, u, proof); err != nil {
		return err
	}
	return l.path.WithProcessLock(ctx, key+"-"+hash(u.IdempotencyID)+".lock", fn)
}

func (l *Ledger) reserve(ctx context.Context, key string, g run.Grant, u run.GrantUse, proof trust.AcceptedProof) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		old, err := l.path.ReadFileLimit(key+".uses", 1<<20)
		var v reservation
		switch {
		case errors.Is(err, os.ErrNotExist):
			old = nil
			v = reservation{"mixlab_artifact_uses_v1", g, proof, nil}
		case err != nil:
			return err
		default:
			if err = json.Unmarshal(old, &v); err != nil {
				return err
			}
			canonical, _ := json.Marshal(v)
			if !bytes.Equal(old, canonical) || v.Format != "mixlab_artifact_uses_v1" || v.Grant != g || v.Proof.Digest != proof.Digest || v.Proof.Generation == 0 || uint64(len(v.Uses)) > g.MaximumUses {
				return fmt.Errorf("conflicting/corrupt grant ledger")
			}
		}
		seen := map[string]bool{}
		found := false
		for _, use := range v.Uses {
			if err := use.Validate(); err != nil {
				return err
			}
			if use.GrantID != g.ID || use.GrantHash != g.SHA256 || use.Operation != g.Operation || use.UploadID != g.UploadID || seen[use.IdempotencyID] {
				return fmt.Errorf("corrupt use binding")
			}
			seen[use.IdempotencyID] = true
			if use.IdempotencyID == u.IdempotencyID {
				if use != u {
					return fmt.Errorf("conflicting use")
				}
				found = true
			}
		}
		if found {
			return nil
		}
		if uint64(len(v.Uses)) >= g.MaximumUses {
			return fmt.Errorf("artifact grant exhausted")
		}
		v.Uses = append(v.Uses, u)
		next, err := json.Marshal(v)
		if err != nil {
			return err
		}
		if len(next) > 1<<20 {
			return fmt.Errorf("grant ledger exceeds bound")
		}
		err = l.path.CompareAndSwap(key+".uses", old, next)
		if errors.Is(err, statehome.ErrConflict) {
			continue
		}
		if err != nil {
			return err
		}
		return nil
	}
}
