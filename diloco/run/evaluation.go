package run

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/precision"
)

type EvaluationAssignment struct {
	Format         string           `json:"format"`
	ID             string           `json:"request_id"`
	Binding        Binding          `json:"binding"`
	Coordinator    Coordinator      `json:"coordinator"`
	Round          uint64           `json:"round"`
	Model          Blob             `json:"model"`
	ValidationHash string           `json:"validation_hash"`
	Split          string           `json:"split"`
	BatchTokens    uint64           `json:"batch_tokens"`
	SequenceLength uint64           `json:"seq_len"`
	Batches        uint64           `json:"batches"`
	WorkerID       string           `json:"worker_id"`
	Precision      precision.Policy `json:"precision"`
	PrecisionHash  string           `json:"precision_hash"`
	IssuedAt       int64            `json:"issued_at"`
	ExpiresAt      int64            `json:"expires_at"`
}

func (a EvaluationAssignment) Validate() error {
	if a.Format != "mixlab_evaluation_assignment_v1" || !wire.Text(a.ID) || !wire.Text(a.WorkerID) || !wire.Text(a.Split) || !wire.Hash(a.ValidationHash) || a.SequenceLength == 0 || a.BatchTokens == 0 || a.BatchTokens%a.SequenceLength != 0 || a.Batches == 0 || !wire.Window(a.IssuedAt, a.ExpiresAt) {
		return fmt.Errorf("invalid global evaluation assignment")
	}
	for _, e := range []error{a.Binding.Validate(), a.Coordinator.Validate(), a.Model.Validate()} {
		if e != nil {
			return e
		}
	}
	h, e := a.Precision.Digest()
	if e != nil {
		return e
	}
	if h != a.PrecisionHash || a.Precision.Backend != "metal" || a.Model.Kind != "model_f32" {
		return fmt.Errorf("evaluation model/policy mismatch")
	}
	return nil
}

type EvaluationResult struct {
	Format            string               `json:"format"`
	Assignment        EvaluationAssignment `json:"assignment"`
	AssignmentHash    string               `json:"assignment_hash"`
	Loss              float64              `json:"loss"`
	EffectiveTokens   uint64               `json:"effective_tokens"`
	StartedAt         int64                `json:"started_at"`
	FinishedAt        int64                `json:"finished_at"`
	PrivateBeforeHash string               `json:"private_before_hash"`
	PrivateAfterHash  string               `json:"private_after_hash"`
}

func (r EvaluationResult) Validate() error {
	if e := r.Assignment.Validate(); e != nil {
		return e
	}
	h, e := Digest(r.Assignment)
	if e != nil {
		return e
	}
	if r.Format != "mixlab_evaluation_result_v1" || r.AssignmentHash != h || !wire.Finite(r.Loss) || r.Loss < 0 || r.EffectiveTokens == 0 || r.StartedAt < r.Assignment.IssuedAt || r.FinishedAt < r.StartedAt || r.FinishedAt > r.Assignment.ExpiresAt || !wire.Hash(r.PrivateBeforeHash) || r.PrivateBeforeHash != r.PrivateAfterHash {
		return fmt.Errorf("invalid evaluation result or mutated private state")
	}
	return nil
}

type EvaluationDecision struct {
	Format     string  `json:"format"`
	Binding    Binding `json:"binding"`
	Round      uint64  `json:"round"`
	ResultHash string  `json:"result_hash"`
	Decision   string  `json:"decision"`
}

func (d EvaluationDecision) Validate() error {
	if d.Format != "mixlab_evaluation_decision_v1" || !wire.Hash(d.ResultHash) || (d.Decision != "continue" && d.Decision != "complete") {
		return fmt.Errorf("invalid evaluation decision")
	}
	return d.Binding.Validate()
}
