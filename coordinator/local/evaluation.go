package local

import (
	"context"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/run"
)

// EvaluationResult reads only the committed chain, never an orphan record
// written before a failed head publication. Results survive subsequent rounds.
func (r *Repository) EvaluationResult(ctx context.Context, id, assignment string) (run.EvaluationResult, error) {
	h, _, err := r.current(id)
	if err != nil {
		return run.EvaluationResult{}, err
	}
	for h.Revision > 0 {
		if err := ctx.Err(); err != nil {
			return run.EvaluationResult{}, err
		}
		v, err := r.read(id, h)
		if err != nil {
			return run.EvaluationResult{}, err
		}
		if v.Change.Evaluation != nil && v.Change.Evaluation.Assignment.ID == assignment {
			return *v.Change.Evaluation, nil
		}
		h = v.Previous
	}
	return run.EvaluationResult{}, coordinator.ErrNotFound
}
