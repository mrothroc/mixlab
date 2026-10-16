package learner

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

type EvaluationReader interface {
	ReadEvaluation(context.Context, run.EvaluationAssignment, io.Writer) error
}
type EvaluateModel func(context.Context, run.EvaluationAssignment, []optimizer.Tensor) (loss float64, tokens uint64, err error)

// Evaluate has no training-engine mutation port. The adapter receives only the
// verified global model and immutable dataset policy. The exact result is kept
// with the completed journal before a caller submits it to the coordinator.
func (r *Runtime) Evaluate(ctx context.Context, a run.EvaluationAssignment, evaluate EvaluateModel) (out run.EvaluationResult, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	reader, ok := r.Peer.(EvaluationReader)
	if r.Store == nil || !ok || evaluate == nil {
		return out, fmt.Errorf("evaluation ports required")
	}
	if err = a.Validate(); err != nil {
		return out, err
	}
	s := r.Scope
	if a.Binding.RunID != s.RunID || a.Binding.PlanHash != s.PlanHash || a.Binding.CohortHash != s.CohortHash || a.Binding.Generation != s.Generation || a.WorkerID != s.WorkerID || a.Coordinator != s.Coordinator {
		return out, fmt.Errorf("evaluation scope mismatch")
	}
	if r.Now == nil {
		r.Now = func() int64 { return time.Now().Unix() }
	}
	err = r.Store.locked(ctx, func() error {
		old, j, e := r.Store.load()
		if e != nil {
			return e
		}
		if j.Completed == nil || j.Assignment.Binding != a.Binding || j.Assignment.Round != a.Round || j.Assignment.WorkerID != a.WorkerID || j.Completed.Precision != a.Precision {
			return fmt.Errorf("evaluation requires same-round completed private state")
		}
		if j.Evaluation != nil {
			if j.Evaluation.Assignment != a {
				return fmt.Errorf("evaluation policy changed")
			}
			out = *j.Evaluation
			return nil
		}
		if r.Now() < a.IssuedAt || r.Now() >= a.ExpiresAt {
			return fmt.Errorf("evaluation deadline expired")
		}
		ctx, cancel := context.WithTimeout(ctx, time.Duration(a.ExpiresAt-r.Now())*time.Second)
		defer cancel()
		if a.Model.Ref.Bytes > r.Store.limit {
			return fmt.Errorf("evaluation model exceeds budget")
		}
		if _, e = r.Store.restore(ctx, *j.Completed); e != nil {
			return e
		}
		var buf bytes.Buffer
		if e = reader.ReadEvaluation(ctx, a, &boundedWriter{dst: &buf, left: a.Model.Ref.Bytes}); e != nil {
			return e
		}
		if e = artifact.Copy(ctx, io.Discard, bytes.NewReader(buf.Bytes()), a.Model.Ref); e != nil {
			return e
		}
		model, e := tensors.Read(bytes.NewReader(buf.Bytes()), a.Model.Ref.Bytes)
		if e != nil {
			return e
		}
		start := r.Now()
		loss, tokens, e := evaluate(ctx, a, model)
		if e != nil {
			return e
		}
		h, e := run.Digest(a)
		if e != nil {
			return e
		}
		out = run.EvaluationResult{Format: "mixlab_evaluation_result_v1", Assignment: a, AssignmentHash: h, Loss: loss, EffectiveTokens: tokens, StartedAt: start, FinishedAt: r.Now(), PrivateBeforeHash: j.Marker.PrivateHash, PrivateAfterHash: j.Marker.PrivateHash}
		if e = out.Validate(); e != nil {
			return e
		}
		current, _, e := r.Store.load()
		if e != nil {
			return e
		}
		if !bytes.Equal(current, old) {
			return fmt.Errorf("evaluation changed private journal")
		}
		j.Evaluation = &out
		return r.Store.save(old, j, "evaluation")
	})
	return
}
