// Package coordinator owns fixed-cohort round decisions. P1 provides the round
// model against ports; durable and HTTP adapters live in child packages, but no
// service is installed. Callers authenticate and authorize before invoking it.
package coordinator

import (
	"context"
	"errors"

	"github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/diloco/run"
)

var (
	ErrNotFound = errors.New("coordinator record not found")
	ErrConflict = errors.New("coordinator concurrent/conflicting operation")
)

// Repository returns independent snapshots. Save atomically compares Revision
// (zero means absent), publishes Next and any receipt/round/evaluation record.
// Receipts survive round advancement; a failed Save changes nothing. Ambiguous
// successful writes are resolved by reloading/retrying the same operation.
type Repository interface {
	Load(context.Context, string) (State, error)
	Receipt(context.Context, string, string) (learner.Receipt, error)
	Save(context.Context, string, uint64, Change) error
}

type Change struct {
	Next       State
	Receipt    *learner.Receipt
	Published  *run.Round
	Evaluation *run.EvaluationResult
	Decision   *run.EvaluationDecision
}

// Artifacts must verify content digest/size, decode bounded float32 tensors in
// program layout order, and return owned arrays. Put is content-addressed and
// idempotent. The local adapter supplies bounded safetensors/streaming storage.
type Artifacts interface {
	Load(context.Context, run.Blob) ([]optimizer.Tensor, error)
	Put(context.Context, string, []optimizer.Tensor) (run.Blob, error)
}

// Ready checks the original allocation is live and authority handoff complete.
// It never recruits replacements. Provider policy stays in the adapter.
type Allocations interface {
	Ready(context.Context, run.Slot) error
}
type Clock interface{ Unix() int64 }
type Evaluator interface {
	Evaluate(context.Context, run.EvaluationAssignment) (run.EvaluationResult, error)
}

type Model struct {
	Repository  Repository
	Artifacts   Artifacts
	Allocations Allocations
	Clock       Clock
	Evaluator   Evaluator
}

type Phase string

const (
	Staged      Phase = "PLAN_STAGED"
	Aborted     Phase = "ADMISSION_ABORTED"
	Ready       Phase = "COHORT_COMMITTED"
	Open        Phase = "ROUND_OPEN"
	Aggregating Phase = "AGGREGATING"
	Publishing  Phase = "PUBLISHING"
	Evaluating  Phase = "EVALUATING"
	Complete    Phase = "COMPLETE"
	Failed      Phase = "FAILED"
)

type EvaluationPolicy struct {
	WorkerID       string   `json:"worker_id"`
	ValidationHash string   `json:"validation_hash"`
	Split          string   `json:"split"`
	BatchTokens    uint64   `json:"batch_tokens"`
	SequenceLength uint64   `json:"seq_len"`
	Batches        uint64   `json:"batches"`
	TargetLoss     *float64 `json:"target_loss"`
	TimeoutSeconds int64    `json:"timeout_seconds"`
}

type State struct {
	Revision      uint64                    `json:"revision"`
	Phase         Phase                     `json:"phase"`
	Plan          run.Plan                  `json:"plan"`
	Commit        *run.Commit               `json:"commit"`
	Policy        EvaluationPolicy          `json:"evaluation_policy"`
	Round         uint64                    `json:"round"`
	Base          run.Blob                  `json:"base"`
	Momentum      run.Blob                  `json:"momentum"`
	PrivateHashes []string                  `json:"private_hashes"`
	Assignments   []run.Assignment          `json:"assignments"`
	Updates       []learner.Update          `json:"updates"`
	Candidate     *run.Round                `json:"candidate"`
	Evaluation    *run.EvaluationAssignment `json:"evaluation"`
	Failure       string                    `json:"failure"`
	Attempts      []run.AttemptBinding      `json:"attempts,omitempty"`
}
