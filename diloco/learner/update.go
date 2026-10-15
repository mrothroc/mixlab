package learner

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/diloco/run"
)

type Update struct {
	Format            string       `json:"format"`
	ID                string       `json:"update_id"`
	AssignmentHash    string       `json:"assignment_hash"`
	Binding           run.Binding  `json:"binding"`
	Round             uint64       `json:"round"`
	BaseRound         uint64       `json:"base_round"`
	BaseModelHash     string       `json:"base_model_hash"`
	WorkerID          string       `json:"worker_id"`
	ParticipantID     string       `json:"participant_id"`
	IslandID          string       `json:"island_id"`
	Contract          run.Contract `json:"contract"`
	Delta             run.Blob     `json:"delta"`
	DType             string       `json:"dtype"`
	TensorCount       uint64       `json:"tensor_count"`
	PrecisionHash     string       `json:"precision_hash"`
	PrivateHash       string       `json:"private_hash"`
	Attempted         uint64       `json:"attempted_steps"`
	Committed         uint64       `json:"committed_steps"`
	Skipped           uint64       `json:"skipped_steps"`
	EffectiveTokens   uint64       `json:"effective_tokens"`
	EffectiveExamples uint64       `json:"effective_examples"`
	Loss              float64      `json:"loss"`
	DeltaL2           float64      `json:"delta_l2"`
	DeltaMaxAbs       float64      `json:"delta_max_abs"`
	StartedAt         int64        `json:"started_at"`
	FinishedAt        int64        `json:"finished_at"`
}

func (u Update) Validate() error {
	if u.Format != "mixlab_outer_update_v1" || !wire.Text(u.ID) || !wire.Text(u.WorkerID) || !wire.Text(u.ParticipantID) || !wire.Text(u.IslandID) || u.Round != u.BaseRound || !wire.Hash(u.AssignmentHash) || !wire.Hash(u.BaseModelHash) || !wire.Hash(u.PrecisionHash) || !wire.Hash(u.PrivateHash) {
		return fmt.Errorf("invalid outer update binding")
	}
	for _, e := range []error{u.Binding.Validate(), u.Contract.Validate(), u.Delta.Validate()} {
		if e != nil {
			return e
		}
	}
	if u.Delta.Kind != "delta_f32" || u.DType != "float32" || u.TensorCount == 0 || u.TensorCount > 1<<20 || u.Attempted == 0 || u.Attempted != u.Committed || u.Skipped != 0 || u.EffectiveTokens == 0 || u.EffectiveExamples == 0 || u.StartedAt <= 0 || u.FinishedAt < u.StartedAt {
		return fmt.Errorf("invalid update counts/artifact")
	}
	for _, v := range []float64{u.Loss, u.DeltaL2, u.DeltaMaxAbs} {
		if !wire.Finite(v) || v < 0 {
			return fmt.Errorf("non-finite/negative update summary")
		}
	}
	return nil
}
func (u Update) ValidateFor(a run.Assignment) error {
	if e := a.Validate(); e != nil {
		return e
	}
	if e := u.Validate(); e != nil {
		return e
	}
	h, e := wire.Digest(a)
	if e != nil {
		return e
	}
	if u.AssignmentHash != h || u.Binding != a.Binding || u.Round != a.Round || u.WorkerID != a.WorkerID || u.Contract != a.Contract || u.BaseModelHash != a.BaseModel.Ref.SHA256 || u.PrecisionHash != a.PrecisionHash || u.Committed != a.InnerSteps || u.StartedAt < a.IssuedAt || u.FinishedAt > a.ExpiresAt {
		return fmt.Errorf("outer update differs from assignment")
	}
	return nil
}

// Submission carries replaceable transport evidence outside semantic updates.
type Submission struct {
	Format             string         `json:"format"`
	Update             Update         `json:"update"`
	PrincipalID        string         `json:"principal_id"`
	AttemptID          string         `json:"attempt_id"`
	WorkloadGeneration uint64         `json:"workload_generation"`
	Allocation         run.Allocation `json:"allocation"`
}

func (s Submission) Validate() error {
	if s.Format != "mixlab_update_submission_v1" || !wire.Text(s.PrincipalID) || !wire.Text(s.AttemptID) || s.Allocation.NodeID != s.Update.ParticipantID {
		return fmt.Errorf("invalid submission envelope")
	}
	if e := s.Allocation.Validate(); e != nil {
		return e
	}
	return s.Update.Validate()
}

type Receipt struct {
	Format      string      `json:"format"`
	Binding     run.Binding `json:"binding"`
	UpdateID    string      `json:"update_id"`
	UpdateHash  string      `json:"update_hash"`
	PrivateHash string      `json:"private_hash"`
}

func (r Receipt) Validate() error {
	if r.Format != "mixlab_update_receipt_v1" || !wire.Text(r.UpdateID) || !wire.Hash(r.UpdateHash) || !wire.Hash(r.PrivateHash) {
		return fmt.Errorf("invalid durable update receipt")
	}
	return r.Binding.Validate()
}
