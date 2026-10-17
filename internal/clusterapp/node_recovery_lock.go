package clusterapp

import (
	"context"
	"time"
)

// Recovery shares the node's serialization fence with longer admission work,
// but must still return within its own HTTP deadline. Never spawn an unowned
// goroutine that can acquire the mutex after the request has gone away.
func (a *NodeJobs) lockRecovery(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if a.mu.TryLock() {
		if err := ctx.Err(); err != nil {
			a.mu.Unlock()
			return err
		}
		return nil
	}
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if a.mu.TryLock() {
				if err := ctx.Err(); err != nil {
					a.mu.Unlock()
					return err
				}
				return nil
			}
		}
	}
}
