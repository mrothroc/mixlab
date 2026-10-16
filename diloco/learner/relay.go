package learner

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/diloco/run"
)

// StartSubmission is replaceable agent evidence, not private replay identity.
type StartSubmission struct {
	Format             string         `json:"format"`
	Start              Start          `json:"start"`
	PrincipalID        string         `json:"principal_id"`
	AttemptID          string         `json:"attempt_id"`
	WorkloadGeneration uint64         `json:"workload_generation"`
	Allocation         run.Allocation `json:"allocation"`
}

func (s StartSubmission) Validate() error {
	if s.Format != "mixlab_start_submission_v1" || !wire.Text(s.PrincipalID) || !wire.Text(s.AttemptID) || s.WorkloadGeneration == 0 {
		return fmt.Errorf("invalid start submission")
	}
	if err := s.Allocation.Validate(); err != nil {
		return err
	}
	return s.Start.Validate()
}

// RelayRequest/Response are safe for local IPC: no keys, certificates or grants.
// Receipt queries carry the original update to bind ownership and exact bytes.
type RelayRequest struct {
	Format     string                `json:"format"`
	Binding    run.Binding           `json:"binding"`
	WorkerID   string                `json:"worker_id"`
	Operation  string                `json:"operation"`
	Start      *Start                `json:"start,omitempty"`
	Update     *Update               `json:"update,omitempty"`
	Evaluation *run.EvaluationResult `json:"evaluation,omitempty"`
	Artifact   *ArtifactRequest      `json:"artifact,omitempty"`
}
type ArtifactRequest struct {
	Assignment run.Assignment            `json:"assignment"`
	Evaluation *run.EvaluationAssignment `json:"evaluation,omitempty"`
	Blob       run.Blob                  `json:"blob"`
	Operation  string                    `json:"operation"`
}

func (q RelayRequest) Validate() error {
	if q.Format != "mixlab_learner_relay_v1" || !wire.Text(q.WorkerID) {
		return fmt.Errorf("invalid relay request")
	}
	if err := q.Binding.Validate(); err != nil {
		return err
	}
	n := 0
	if q.Start != nil {
		n++
		if err := q.Start.Validate(); err != nil {
			return err
		}
		if q.Start.Binding != q.Binding || q.Start.WorkerID != q.WorkerID {
			return fmt.Errorf("start scope mismatch")
		}
	}
	if q.Update != nil {
		n++
		if err := q.Update.Validate(); err != nil {
			return err
		}
		if q.Update.Binding != q.Binding || q.Update.WorkerID != q.WorkerID {
			return fmt.Errorf("update scope mismatch")
		}
	}
	if q.Evaluation != nil {
		n++
		if err := q.Evaluation.Validate(); err != nil {
			return err
		}
		if q.Evaluation.Assignment.Binding != q.Binding || q.Evaluation.Assignment.WorkerID != q.WorkerID {
			return fmt.Errorf("evaluation scope mismatch")
		}
	}
	if q.Artifact != nil {
		n++
		a := q.Artifact
		binding, worker := a.Assignment.Binding, a.Assignment.WorkerID
		if a.Evaluation != nil {
			if a.Assignment != (run.Assignment{}) || a.Operation != "read" || a.Blob != a.Evaluation.Model {
				return fmt.Errorf("evaluation artifact must read only its assigned model")
			}
			if err := a.Evaluation.Validate(); err != nil {
				return err
			}
			binding, worker = a.Evaluation.Binding, a.Evaluation.WorkerID
		} else if err := a.Assignment.Validate(); err != nil {
			return err
		}
		if err := a.Blob.Validate(); err != nil {
			return err
		}
		if binding != q.Binding || worker != q.WorkerID || (a.Operation != "read" && a.Operation != "create") {
			return fmt.Errorf("artifact scope mismatch")
		}
	}
	if (q.Operation == "poll" && n == 0) || (n == 1 && ((q.Operation == "start" && q.Start != nil) || ((q.Operation == "submit" || q.Operation == "receipt") && q.Update != nil) || (q.Operation == "evaluate" && q.Evaluation != nil) || (q.Operation == "artifact" && q.Artifact != nil))) {
		return nil
	}
	return fmt.Errorf("invalid relay operation/payload")
}

type RelayResponse struct {
	Format     string                    `json:"format"`
	Assignment *run.Assignment           `json:"assignment,omitempty"`
	Evaluation *run.EvaluationAssignment `json:"evaluation,omitempty"`
	Receipt    *Receipt                  `json:"receipt,omitempty"`
	Done       bool                      `json:"done"`
}

func (r RelayResponse) Validate() error {
	if r.Format != "mixlab_learner_reply_v1" {
		return fmt.Errorf("invalid relay reply")
	}
	n := 0
	if r.Assignment != nil {
		n++
		if err := r.Assignment.Validate(); err != nil {
			return err
		}
	}
	if r.Evaluation != nil {
		n++
		if err := r.Evaluation.Validate(); err != nil {
			return err
		}
	}
	if r.Receipt != nil {
		n++
		if err := r.Receipt.Validate(); err != nil {
			return err
		}
	}
	if n > 1 || (r.Done && n != 0) {
		return fmt.Errorf("ambiguous relay reply")
	}
	return nil
}
