package learner

import (
	"fmt"
	"github.com/mrothroc/mixlab/diloco/internal/wire"
	"github.com/mrothroc/mixlab/diloco/run"
)

// Start commits the identity of a locally durable start snapshot before data
// consumption. It is distinct from the prior round's completed private state.
type Start struct {
	Format         string      `json:"format"`
	Binding        run.Binding `json:"binding"`
	WorkerID       string      `json:"worker_id"`
	AssignmentHash string      `json:"assignment_hash"`
	PrivateHash    string      `json:"private_hash"`
}

func (s Start) Validate() error {
	if s.Format != "mixlab_learner_start_v1" || !wire.Text(s.WorkerID) || !wire.Hash(s.AssignmentHash) || !wire.Hash(s.PrivateHash) {
		return fmt.Errorf("invalid learner start commitment")
	}
	return s.Binding.Validate()
}
