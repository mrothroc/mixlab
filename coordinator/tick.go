package coordinator

import (
	"context"
	"fmt"
)

// Tick makes at most one service-driven transition. Evaluation results arrive
// through the authenticated learner gateway; a background tick must never run
// an evaluator or invent a result. Deadlines survive service restart.
func (m Model) Tick(ctx context.Context, id string, roundSeconds int64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s, err := m.load(ctx, id)
	if err != nil {
		return err
	}
	switch s.Phase {
	case Staged, Aborted, Complete, Failed:
		return nil
	case Ready:
		_, err = m.OpenRound(ctx, id, roundSeconds)
		return err
	case Open:
		if m.Clock.Unix() >= s.Assignments[0].ExpiresAt {
			return m.Fail(ctx, id, "learner round deadline expired")
		}
		if len(s.Updates) == len(s.Plan.Slots) {
			return m.Advance(ctx, id)
		}
		return nil
	case Aggregating, Publishing:
		return m.Advance(ctx, id)
	case Evaluating:
		if m.Clock.Unix() >= s.Evaluation.ExpiresAt {
			return m.Fail(ctx, id, "global evaluation deadline expired")
		}
		return nil
	default:
		return fmt.Errorf("unsupported coordinator phase %q", s.Phase)
	}
}
