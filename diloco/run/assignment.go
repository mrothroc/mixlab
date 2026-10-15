package run

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
)

const AssignmentVersion = "mixlab_round_assignment_v1"

type Assignment struct {
	Format              string      `json:"format"`
	ID                  string      `json:"assignment_id"`
	Binding             Binding     `json:"binding"`
	Coordinator         Coordinator `json:"coordinator"`
	WorkerID            string      `json:"worker_id"`
	Round               uint64      `json:"round"`
	BaseRound           uint64      `json:"base_round"`
	BaseModel           Blob        `json:"base_model"`
	OuterState          Blob        `json:"outer_state"`
	Contract            Contract    `json:"contract"`
	PartitionID         string      `json:"dataset_partition_id"`
	InnerSteps          uint64      `json:"inner_steps"`
	PreviousPrivateHash string      `json:"previous_private_hash"`
	PrecisionHash       string      `json:"precision_hash"`
	IssuedAt            int64       `json:"issued_at"`
	ExpiresAt           int64       `json:"expires_at"`
}

func (a Assignment) Validate() error {
	if a.Format != AssignmentVersion || !wire.Text(a.ID) || !wire.Text(a.WorkerID) || !wire.Text(a.PartitionID) || a.InnerSteps == 0 || a.Round != a.BaseRound || !wire.Hash(a.PrecisionHash) || !wire.Window(a.IssuedAt, a.ExpiresAt) {
		return fmt.Errorf("invalid round assignment")
	}
	if (a.Round > 0 || a.PreviousPrivateHash != "") && !wire.Hash(a.PreviousPrivateHash) {
		return fmt.Errorf("previous private state required after round zero")
	}
	for _, e := range []error{a.Binding.Validate(), a.Coordinator.Validate(), a.Contract.Validate(), a.BaseModel.Validate(), a.OuterState.Validate()} {
		if e != nil {
			return e
		}
	}
	if a.BaseModel.Kind != "model_f32" || a.OuterState.Kind != "outer_state_f32" {
		return fmt.Errorf("assignment artifact kind mismatch")
	}
	return nil
}
func (a Assignment) ValidateFor(p Plan, c Commit) error {
	if e := c.ValidateFor(p); e != nil {
		return e
	}
	if e := a.Validate(); e != nil {
		return e
	}
	if a.Binding != c.Binding || a.Coordinator != c.Coordinator || a.Contract != p.Contract || a.InnerSteps != p.Outer.InnerSteps || a.Round >= p.Outer.TotalSteps/p.Outer.InnerSteps || a.IssuedAt < c.CommittedAt {
		return fmt.Errorf("assignment differs from committed run")
	}
	if a.Round == 0 && a.BaseModel != p.BaseModel {
		return fmt.Errorf("initial model mismatch")
	}
	for _, r := range c.Registrations {
		if r.Slot.Member.WorkerID == a.WorkerID && r.Slot.Member.DatasetPartitionID == a.PartitionID && r.PrecisionHash == a.PrecisionHash {
			return nil
		}
	}
	return fmt.Errorf("assignment worker/policy mismatch")
}

// AttemptBinding is operational fencing evidence, intentionally not part of
// Assignment or RNG identity. Node management must authorize this separately.
type AttemptBinding struct {
	Format                string     `json:"format"`
	Binding               Binding    `json:"binding"`
	WorkerID              string     `json:"worker_id"`
	PreviousAttemptID     string     `json:"previous_attempt_id"`
	AttemptID             string     `json:"attempt_id"`
	WorkloadGeneration    uint64     `json:"workload_generation"`
	Allocation            Allocation `json:"allocation"`
	NodeAuthorizationHash string     `json:"node_authorization_hash"`
	PrivateHash           string     `json:"private_hash"`
	PrecisionHash         string     `json:"precision_hash"`
}

func (a AttemptBinding) Validate() error {
	if a.Format != "mixlab_learner_attempt_v1" || !wire.Text(a.WorkerID) || !wire.Text(a.PreviousAttemptID) || !wire.Text(a.AttemptID) || a.AttemptID == a.PreviousAttemptID || !wire.Hash(a.NodeAuthorizationHash) || !wire.Hash(a.PrivateHash) || !wire.Hash(a.PrecisionHash) {
		return fmt.Errorf("invalid fenced retry binding")
	}
	if e := a.Binding.Validate(); e != nil {
		return e
	}
	return a.Allocation.Validate()
}

// ValidateFor checks immutable recovery bindings. Node authorization, old
// process fencing and allocation liveness remain mandatory external checks.
func (a AttemptBinding) ValidateFor(c Commit, previousAttempt string, previousGeneration uint64, expectedPrivate string) error {
	if e := c.Validate(); e != nil {
		return e
	}
	if e := a.Validate(); e != nil {
		return e
	}
	if a.Binding != c.Binding || a.PreviousAttemptID != previousAttempt || a.WorkloadGeneration <= previousGeneration || a.PrivateHash != expectedPrivate {
		return fmt.Errorf("retry differs from expected prior attempt/private state")
	}
	for _, r := range c.Registrations {
		if r.Slot.Member.WorkerID == a.WorkerID && r.Slot.Allocation == a.Allocation && r.PrecisionHash == a.PrecisionHash {
			return nil
		}
	}
	return fmt.Errorf("retry cannot replace allocation or precision policy")
}
