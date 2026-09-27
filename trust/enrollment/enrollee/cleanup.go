package enrollee

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/mrothroc/mixlab/trust/keylifecycle"
)

// Abort destroys only this provisional context's exact keys and then removes
// its staging directory. It refuses any attempted/published credential; those
// identities require explicit reconciliation, never automatic key deletion.
func (c *Client) Abort(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.published {
		return fmt.Errorf("credential publication attempted; inspect final/staging identity before cleanup")
	}
	if _, err := os.Lstat(c.final.Dir()); !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("final identity exists or is inaccessible; refusing provisional cleanup")
	}
	return c.stage.WithProcessLock(ctx, operationLock, func() error {
		if _, err := os.Lstat(c.final.Dir()); !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("final identity appeared; refusing cleanup")
		}
		if _, err := c.stage.ReadFileLimit("principal.json", 1<<20); !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("staging contains a credential or cannot be inspected; explicit reconciliation required")
		}
		for _, slot := range []keylifecycle.Slot{keylifecycle.Principal, keylifecycle.NodeEnvelope} {
			if slot == keylifecycle.NodeEnvelope && string(c.intent.Role) != "node" {
				continue
			}
			r, err := c.manager.View(slot)
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			if err != nil {
				return err
			}
			switch r.Stage {
			case "active":
				_, err = c.manager.Destroy(ctx, slot, r.Active.ID)
			case "creating":
				_, err = c.manager.Abort(ctx, slot, r.Candidate.ID)
			case "destroying", "aborting":
				_, err = c.manager.Recover(ctx, slot)
			case "destroyed", "aborted":
			default:
				return fmt.Errorf("unexpected provisional key lifecycle %s", r.Stage)
			}
			if err != nil {
				return err
			}
		}
		if err := c.keys.Close(); err != nil {
			return err
		}
		if err := c.stage.Validate(); err != nil {
			return err
		}
		if err := os.RemoveAll(c.stage.Dir()); err != nil {
			return err
		}
		parent, err := os.Open(filepath.Dir(c.stage.Dir()))
		if err != nil {
			return err
		}
		defer func() { _ = parent.Close() }()
		return parent.Sync()
	})
}
