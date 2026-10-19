package learnerapi

import (
	"context"
	"net/http"

	"github.com/mrothroc/mixlab/coordinator"
)

func (g Gateway) authorizedState(ctx context.Context, r *http.Request, q Request, id string) (coordinator.State, error) {
	s, err := g.Model.Repository.Load(ctx, id)
	if err != nil {
		return s, err
	}
	err = g.authorize(ctx, r, q, s)
	if err == nil || !q.Call.TerminalRead() || s.Phase == coordinator.Complete {
		return s, err
	}
	// Allocation readiness may observe COMPLETE after the initial state read.
	// Reauthorize from scratch, and return that same fresh state, never the
	// obsolete assignment whose allocation check failed. No live-run fallback.
	latest, readErr := g.Model.Repository.Load(ctx, id)
	if readErr == nil && latest.Phase == coordinator.Complete {
		return latest, g.authorize(ctx, r, q, latest)
	}
	return s, err
}
